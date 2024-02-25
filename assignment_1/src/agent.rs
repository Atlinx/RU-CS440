use owo_colors::OwoColorize;
use std::{
    collections::{HashSet, VecDeque},
    fmt::Debug,
    rc::Rc,
    sync::Mutex,
};

use crate::prelude::*;

// region AgentBehavior
pub trait AgentBehavior: Debug {
    /// Initializes the behavior
    fn init(&mut self, _map_size: Vec2i) {}
    /// Attempts to find a path from the agent's current position to the goal
    ///
    /// Returns (path from self to goal in reversed order (first = goal, last = path), number of expanded cells)
    fn pathfind(&mut self, agent: &Agent) -> (VecDeque<Vec2i>, i32);
    /// Prints a representation of the agent's behavior
    fn map_str(&self, agent: &Agent, prev_agent_pos: Vec2i) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(agent.mind_map.size().x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        let agent_path_hashset: HashSet<Vec2i> = agent.target_path.clone().into_iter().collect();
        for y in 0..agent.mind_map.size().y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..agent.mind_map.size().x {
                let pos = Vec2i::new(x, y);
                if agent.mind_map.get_cell(pos).unwrap() {
                    str += &format!("{}", "XX".on_truecolor(125, 125, 125));
                } else {
                    if prev_agent_pos == pos {
                        str +=
                            &format!("{}", "AA".truecolor(172, 240, 46).on_truecolor(69, 163, 65));
                    } else if agent.goal == pos {
                        str += &format!(
                            "{}",
                            "GG".truecolor(240, 227, 46).on_truecolor(230, 137, 39)
                        );
                    } else if agent_path_hashset.contains(&pos) || pos == agent.position {
                        str += &format!("{}", "pp".cyan());
                    } else {
                        str += "  ";
                    }
                }
            }
            str += &format!("{}\n", "XX".on_truecolor(125, 125, 125));
        }
        str += &format!(
            "{}\n",
            "XX".repeat(agent.mind_map.size().x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        str
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BreakTieMode {
    HigherGCost,
    None,
    LowerGCost,
}

#[derive(Debug, Clone)]
pub struct AStarBehavior {
    pub is_forward: bool,
    pub break_tie_mode: BreakTieMode,
}
impl AStarBehavior {
    pub fn new(is_forward: bool, break_tie_mode: BreakTieMode) -> AStarBehavior {
        AStarBehavior {
            is_forward,
            break_tie_mode,
        }
    }
    /// Returns path from start to end, excluding start
    /// Given in stack order (end position is first, start position is last)
    fn astar_pathfind(
        wall_map: &WallMap,
        mut start: Vec2i,
        mut end: Vec2i,
        break_tie_mode: BreakTieMode,
        is_forward: bool,
    ) -> (VecDeque<Vec2i>, i32) {
        let mut open_states = BinaryHeap::<Rc<AStarState>>::new();
        let mut closed_cells = HashSet::<Vec2i>::new();

        if !is_forward {
            let temp = start;
            start = end;
            end = temp;
        }

        let mut expanded_cell_count = 0;
        let mut age = 0;
        open_states.push(Rc::new(AStarState::from_g_h_costs(
            0,
            start.manhattan_distance(&end),
            start,
            None,
            break_tie_mode,
            age,
        )));
        while let Some(curr_state) = open_states.pop() {
            if closed_cells.contains(&curr_state.cell) {
                // There was an earlier path that beat us to the punch.
                // This state is no longer relevant.
                continue;
            }
            expanded_cell_count += 1;
            closed_cells.insert(curr_state.cell);
            if curr_state.cell == end {
                // Path has been found
                let mut path = VecDeque::new();
                path.push_back(curr_state.cell);
                let mut curr_state_ref = &curr_state;
                // Create path by walking up the parent chain
                while let Some(parent_state) = curr_state_ref.parent_state.as_ref() {
                    if is_forward {
                        path.push_front(parent_state.cell);
                    } else {
                        path.push_back(parent_state.cell);
                    }
                    curr_state_ref = parent_state;
                }
                return (path, expanded_cell_count);
            }
            for (neighbor_cell, neighbor_filled) in
                wall_map.get_neighbor_cells_4_way(curr_state.cell)
            {
                // Only consider neighbors that are empty and are not explored yet
                if !neighbor_filled && !closed_cells.contains(&neighbor_cell) {
                    age += 1;
                    open_states.push(Rc::new(AStarState::from_g_h_costs(
                        curr_state.actual_cost + 1,             // Cost for every move is 1
                        neighbor_cell.manhattan_distance(&end), // Heuristic is the manhattan distance between the cell and the goal
                        neighbor_cell,
                        Some(curr_state.clone()),
                        break_tie_mode,
                        age,
                    )));
                }
            }
        }
        // No path found
        (VecDeque::new(), expanded_cell_count)
    }
}
impl AgentBehavior for AStarBehavior {
    fn pathfind(&mut self, agent: &Agent) -> (VecDeque<Vec2i>, i32) {
        let (mut path, expanded_cell_count) = Self::astar_pathfind(
            &agent.mind_map,
            agent.position,
            agent.goal,
            self.break_tie_mode,
            self.is_forward,
        );
        // No need to include the starting positioon
        path.pop_front();
        (path, expanded_cell_count)
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveAStarBehavior {
    pub break_tie_mode: BreakTieMode,
    pub actual_cost_map: Map<i32>,
    pub h_cost_map: Map<i32>,
    pub f_cost_map: Map<i32>,
    pub debug: bool,
}
impl AdaptiveAStarBehavior {
    pub fn new(break_tie_mode: BreakTieMode, debug: bool) -> AdaptiveAStarBehavior {
        AdaptiveAStarBehavior {
            break_tie_mode,
            debug,
            actual_cost_map: Map::<i32>::new(Vec2i::ONE),
            h_cost_map: Map::<i32>::new(Vec2i::ONE),
            f_cost_map: Map::<i32>::new(Vec2i::ONE),
        }
    }
    fn calc_heuristic_cost(&self, agent: &Agent, position: Vec2i) -> i32 {
        let cached_actual_cost = self.actual_cost_map.get_cell(position).unwrap();
        if cached_actual_cost >= 0 {
            assert!(
                self.actual_goal_cost(agent) >= 0,
                "Expected goal cost to exist"
            );
            // h(s) = g(s_goal) - g(s)
            self.actual_goal_cost(agent) - cached_actual_cost
        } else {
            position.manhattan_distance(&agent.goal)
        }
    }
    fn actual_goal_cost(&self, agent: &Agent) -> i32 {
        self.actual_cost_map.get_cell(agent.goal).unwrap()
    }
    fn cost_map_str(
        cost_map: &Map<i32>,
        agent: &Agent,
        prev_agent_pos: Vec2i,
        color: (u8, u8, u8),
    ) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(agent.mind_map.size().x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        let agent_path_hashset: HashSet<Vec2i> = agent.target_path.clone().into_iter().collect();
        for y in 0..agent.mind_map.size().y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..agent.mind_map.size().x {
                let pos = Vec2i::new(x, y);
                if agent.mind_map.get_cell(pos).unwrap() {
                    str += &format!("{}", "XX".on_truecolor(125, 125, 125));
                } else {
                    if prev_agent_pos == pos {
                        str +=
                            &format!("{}", "AA".truecolor(172, 240, 46).on_truecolor(69, 163, 65));
                    } else if agent.goal == pos {
                        str += &format!(
                            "{}",
                            "GG".truecolor(240, 227, 46).on_truecolor(230, 137, 39)
                        );
                    } else {
                        let cost = cost_map.get_cell(pos).unwrap();
                        // Set cell string
                        let cell_str = {
                            if cost >= 0 {
                                if cost <= 99 {
                                    format!("{: >2}", cost)
                                } else {
                                    "..".to_owned()
                                }
                            } else {
                                "  ".to_owned()
                            }
                        };

                        fn get_colored_tile(
                            cell_str: String,
                            position: Vec2i,
                            fg_1: (u8, u8, u8),
                            fg_2: (u8, u8, u8),
                            bg_1: (u8, u8, u8),
                            bg_2: (u8, u8, u8),
                        ) -> String {
                            if position.y % 2 == 0 {
                                if position.x % 2 == 1 {
                                    format!(
                                        "{}",
                                        cell_str
                                            .truecolor(fg_1.0, fg_1.1, fg_1.2)
                                            .on_truecolor(bg_1.0, bg_1.1, bg_1.2)
                                    )
                                } else {
                                    format!(
                                        "{}",
                                        cell_str
                                            .truecolor(fg_2.0, fg_2.1, fg_2.2)
                                            .on_truecolor(bg_2.0, bg_2.1, bg_2.2)
                                    )
                                }
                            } else {
                                if position.x % 2 == 0 {
                                    format!(
                                        "{}",
                                        cell_str
                                            .truecolor(fg_1.0, fg_1.1, fg_1.2)
                                            .on_truecolor(bg_1.0, bg_1.1, bg_1.2)
                                    )
                                } else {
                                    format!(
                                        "{}",
                                        cell_str
                                            .truecolor(fg_2.0, fg_2.1, fg_2.2)
                                            .on_truecolor(bg_2.0, bg_2.1, bg_2.2)
                                    )
                                }
                            }
                        }

                        let cell_on_path =
                            agent_path_hashset.contains(&pos) || pos == agent.position;

                        // Checkerboard BG
                        str += &{
                            if cell_on_path {
                                // Checkboard Blue FG Blue BG
                                get_colored_tile(
                                    cell_str,
                                    pos,
                                    (97, 142, 255),
                                    (97, 142, 255),
                                    (54, 62, 181),
                                    (36, 41, 115),
                                )
                            } else {
                                // Checkboard Red FG Black BG
                                get_colored_tile(
                                    cell_str,
                                    pos,
                                    color,
                                    color,
                                    (69, 65, 65),
                                    (0, 0, 0),
                                )
                            }
                        };
                    }
                }
            }
            str += &format!("{}\n", "XX".on_truecolor(125, 125, 125));
        }
        str += &format!(
            "{}\n",
            "XX".repeat(agent.mind_map.size().x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        str
    }
}
impl AgentBehavior for AdaptiveAStarBehavior {
    fn init(&mut self, map_size: Vec2i) {
        self.actual_cost_map.resize(map_size);
        self.actual_cost_map.fill(-1);
        if self.debug {
            self.h_cost_map.resize(map_size);
            self.h_cost_map.fill(-1);
            self.f_cost_map.resize(map_size);
            self.f_cost_map.fill(-1);
        }
    }
    fn pathfind(&mut self, agent: &Agent) -> (VecDeque<Vec2i>, i32) {
        let mut open_states = BinaryHeap::<Rc<AStarState>>::new();
        let mut closed_cells = HashSet::<Vec2i>::new();

        let mut age = 0;
        let mut expanded_cell_count = 0;
        open_states.push(Rc::new(AStarState::from_g_h_costs(
            0,
            self.calc_heuristic_cost(agent, agent.position),
            agent.position,
            None,
            self.break_tie_mode,
            age,
        )));
        let mut expanded_states = Vec::new();
        // No path found
        let mut result = (VecDeque::new(), expanded_cell_count);
        while let Some(curr_state) = open_states.pop() {
            if closed_cells.contains(&curr_state.cell) {
                // There was an earlier path that beat us to the punch.
                // This state is no longer relevant.
                continue;
            }
            expanded_cell_count += 1;
            closed_cells.insert(curr_state.cell);
            if curr_state.cell == agent.goal {
                // Path has been found
                let mut path = VecDeque::new();
                path.push_front(curr_state.cell);
                let mut curr_state_ref = &curr_state;
                // Create path by walking up the parent chain
                while let Some(parent_state) = curr_state_ref.parent_state.as_ref() {
                    path.push_front(parent_state.cell);
                    curr_state_ref = parent_state;
                }
                // No need to include the starting positioon
                path.pop_front();
                result = (path, expanded_cell_count);
                expanded_states.push(curr_state);
                break;
            }
            for (neighbor_cell, neighbor_filled) in
                agent.mind_map.get_neighbor_cells_4_way(curr_state.cell)
            {
                // Only consider neighbors that are empty and are not explored yet
                if !neighbor_filled && !closed_cells.contains(&neighbor_cell) {
                    age += 1;
                    let state = AStarState::from_g_h_costs(
                        curr_state.actual_cost + 1, // Cost for every move is 1
                        self.calc_heuristic_cost(agent, neighbor_cell),
                        neighbor_cell,
                        Some(curr_state.clone()),
                        self.break_tie_mode,
                        age,
                    );
                    open_states.push(Rc::new(state));
                }
            }
            expanded_states.push(curr_state);
        }

        self.actual_cost_map.fill(-1);
        if self.debug {
            self.h_cost_map.fill(-1);
            self.f_cost_map.fill(-1);
        }

        for curr_state in expanded_states {
            self.actual_cost_map
                .set_cell(curr_state.cell, curr_state.actual_cost)
                .unwrap();
            if self.debug {
                self.h_cost_map
                    .set_cell(curr_state.cell, curr_state.heuristic_cost)
                    .unwrap();
                self.f_cost_map
                    .set_cell(curr_state.cell, curr_state.f_cost())
                    .unwrap();
            }
        }
        result
    }
    fn map_str(&self, agent: &Agent, prev_agent_pos: Vec2i) -> String {
        if self.debug {
            let f_cost_str = AdaptiveAStarBehavior::cost_map_str(
                &self.f_cost_map,
                agent,
                prev_agent_pos,
                (255, 31, 31),
            );
            let g_cost_str = AdaptiveAStarBehavior::cost_map_str(
                &self.actual_cost_map,
                agent,
                prev_agent_pos,
                (255, 147, 31),
            );
            let h_cost_str = AdaptiveAStarBehavior::cost_map_str(
                &self.h_cost_map,
                agent,
                prev_agent_pos,
                (255, 221, 31),
            );
            f_cost_str
                .join_sides_spaced(&g_cost_str)
                .join_sides_spaced(&h_cost_str)
        } else {
            AdaptiveAStarBehavior::cost_map_str(
                &self.actual_cost_map,
                agent,
                prev_agent_pos,
                (255, 147, 31),
            )
        }
    }
}
// endregion

// region Agent
#[derive(Debug, PartialEq, Eq)]
pub enum AgentStatus {
    Inactive,
    Running,
    Complete(bool),
}

#[derive(Debug)]
pub struct Agent {
    /// Agent's mental map of the environment
    pub mind_map: WallMap,
    /// Current possition of the agent
    pub position: Vec2i,
    /// Goal position the agent wants to reach
    pub goal: Vec2i,
    /// Path to target.
    /// Stored in order from goal to target (reverse order), to promote fast popping
    pub target_path: VecDeque<Vec2i>,
    /// Status of the agent.
    /// - Inactive when the agent is first created
    /// - Running after the agent is initialized
    /// - Complete after the agent finishes path finding (Either the goal was reached or no path can be found)
    pub status: AgentStatus,
    /// Behavior of the agent
    pub behavior: Mutex<Box<dyn AgentBehavior>>,
    /// Total number of cells expanded by the agent over it's entire lifetime
    pub expanded_cell_count: i32,
    /// Number of cells expanded by the agent in the last step
    pub step_expanded_cell_count: i32,
}
#[derive(Clone, Debug)]
struct AStarState {
    pub heap_cost: i32,
    pub actual_cost: i32,
    pub heuristic_cost: i32,
    pub cell: Vec2i,
    pub parent_state: Option<Rc<AStarState>>,
    pub age: u64,
}
impl AStarState {
    // Scaling factor used to break ties.
    const F_SCALE: i32 = 46_340;
    // Maximum supported map width. At this width and height, the maximum
    // distance between the corners of the map = F_SCALE
    const MAX_MAP_WIDTH: i32 = Self::F_SCALE / 2;
    pub fn from_g_h_costs(
        actual_cost: i32,
        heuristic_cost: i32,
        cell: Vec2i,
        parent_state: Option<Rc<AStarState>>,
        break_tie_mode: BreakTieMode,
        age: u64,
    ) -> AStarState {
        let cost = match break_tie_mode {
            BreakTieMode::HigherGCost => {
                Self::F_SCALE * (actual_cost + heuristic_cost) - actual_cost
            }
            BreakTieMode::None => actual_cost + heuristic_cost,
            BreakTieMode::LowerGCost => {
                Self::F_SCALE * (actual_cost + heuristic_cost) - (Self::F_SCALE - actual_cost)
            }
        };
        AStarState {
            cell,
            actual_cost,
            heuristic_cost,
            // Convert the max heap into a min heap.
            heap_cost: -cost,
            parent_state,
            age,
        }
    }
    pub fn f_cost(&self) -> i32 {
        self.actual_cost + self.heuristic_cost
    }
}
impl Eq for AStarState {}
impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.heap_cost == other.heap_cost
    }
}
impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.heap_cost
            .cmp(&other.heap_cost)
            .then(self.age.cmp(&other.age))
    }
}
impl Agent {
    pub fn new(start_pos: Vec2i, goal: Vec2i, behavior: Box<dyn AgentBehavior>) -> Agent {
        Agent {
            expanded_cell_count: 0,
            step_expanded_cell_count: 0,
            position: start_pos,
            goal,
            target_path: VecDeque::new(),
            status: AgentStatus::Inactive,
            mind_map: WallMap::new(Vec2i::ONE),
            behavior: Mutex::new(behavior),
        }
    }
    pub fn init(&mut self, map_size: Vec2i) {
        self.mind_map.resize(map_size);
        self.behavior.try_lock().unwrap().init(map_size);
        self.status = AgentStatus::Running;
    }
    pub fn step(&mut self, world_map: &WallMap) {
        assert!(
            self.status == AgentStatus::Running,
            "Can only step an agent once it's running!"
        );
        if self.position == self.goal {
            // We are already at the goal
            self.status = AgentStatus::Complete(true);
            return;
        }
        // If we don't have a path yet, we should calculate the path
        let mut path_needs_update = self.target_path.len() == 0;
        for dir in Vec2i::DIRECTIONS_4_WAY {
            let neighbor_pos = self.position + dir;
            if let Ok(neighbor_value) = world_map.get_cell(neighbor_pos) {
                self.mind_map
                    .set_cell(neighbor_pos, neighbor_value)
                    .unwrap();
                if let Some(next_pos) = self.target_path.front() {
                    if *next_pos == neighbor_pos && neighbor_value {
                        // If the neighbor is blocked off (true),
                        // and this neighbor is on the next path to the goal
                        // then we need to update the path
                        path_needs_update = true;
                    }
                }
            }
        }
        if path_needs_update {
            // Only regenerate the path if an update is needed
            let (path, expanded_cell_count) = self.behavior.try_lock().unwrap().pathfind(self);
            self.step_expanded_cell_count = expanded_cell_count;
            self.expanded_cell_count += expanded_cell_count;
            self.target_path = path;
        } else {
            self.step_expanded_cell_count = 0;
        }
        if let Some(new_pos) = self.target_path.pop_front() {
            // We have a path, so we move towards the path
            self.position = new_pos;
            if new_pos == self.goal {
                // After moving, we reached the position
                self.status = AgentStatus::Complete(true);
            }
        } else {
            // We couldn't generate a path, so we failed to reach the goal.
            self.status = AgentStatus::Complete(false);
        }
    }
    pub fn mind_map_str(&self, prev_agent_pos: Vec2i) -> String {
        let behavior = self.behavior.try_lock().unwrap();
        behavior.map_str(self, prev_agent_pos)
    }
}
// endregion
