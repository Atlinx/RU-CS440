#![allow(dead_code)]
use owo_colors::OwoColorize;
use rand::{seq::SliceRandom, Rng};
use std::{
    collections::{BinaryHeap, HashSet},
    env,
    fmt::{format, Display},
    hash::Hash,
    ops,
    rc::Rc,
    thread::sleep,
    time::Duration,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let mut size = Vec2i::new(51, 51);
    if let Some(x_str) = args.get(1) {
        if let Ok(x) = x_str.parse() {
            if let Some(y_str) = args.get(2) {
                if let Ok(y) = y_str.parse() {
                    size = Vec2i::new(x, y);
                }
            }
        }
    }

    let map_gen = MazeGenerator::new(size);
    let rand_map = map_gen.generate_dfs_map_default();
    let simulation = SimulationBuilder::from_map(rand_map);
    let mut runner = SimulationRunner::new(simulation, 0.25, true);
    runner.run();
}

// region Vec2i
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Vec2i {
    pub x: i32,
    pub y: i32,
}
impl Vec2i {
    pub const ZERO: Vec2i = Vec2i { x: 0, y: 0 };
    pub const UP: Vec2i = Vec2i { x: 0, y: 1 };
    pub const DOWN: Vec2i = Vec2i { x: 0, y: -1 };
    pub const LEFT: Vec2i = Vec2i { x: -1, y: 0 };
    pub const RIGHT: Vec2i = Vec2i { x: 1, y: 0 };
    pub const DIRECTIONS_4_WAY: [Vec2i; 4] = [Vec2i::UP, Vec2i::DOWN, Vec2i::LEFT, Vec2i::RIGHT];

    pub fn new(x: i32, y: i32) -> Vec2i {
        Vec2i { x, y }
    }
    pub fn manhattan_distance(&self, other: &Self) -> i32 {
        (self.x - other.x).abs() + (self.y - other.y).abs()
    }
}
impl ops::Add<Vec2i> for Vec2i {
    type Output = Vec2i;
    fn add(self, rhs: Vec2i) -> Self::Output {
        Vec2i::new(self.x + rhs.x, self.y + rhs.y)
    }
}
impl ops::Neg for Vec2i {
    type Output = Vec2i;
    fn neg(self) -> Self::Output {
        Vec2i::new(-self.x, -self.y)
    }
}
impl ops::Sub<Vec2i> for Vec2i {
    type Output = Vec2i;
    fn sub(self, rhs: Vec2i) -> Self::Output {
        Vec2i::new(self.x - rhs.x, self.y - rhs.y)
    }
}
impl ops::Div<i32> for Vec2i {
    type Output = Vec2i;
    fn div(self, rhs: i32) -> Self::Output {
        Vec2i::new(self.x / rhs, self.y / rhs)
    }
}
impl ops::Mul<i32> for Vec2i {
    type Output = Vec2i;
    fn mul(self, rhs: i32) -> Self::Output {
        Vec2i::new(self.x * rhs, self.y * rhs)
    }
}

impl Display for Vec2i {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec2({}, {})", self.x, self.y)
    }
}
// endregion

// region Map
#[derive(Debug)]
pub struct Map {
    cells: Vec<bool>,
    size: Vec2i,
}

impl Map {
    pub fn new(size: Vec2i) -> Map {
        Map {
            cells: vec![false; (size.x * size.y) as usize],
            size,
        }
    }
    pub fn size(&self) -> &Vec2i {
        &self.size
    }
    pub fn cells(&self) -> &Vec<bool> {
        &self.cells
    }
    pub fn resize(&mut self, size: Vec2i) {
        self.cells.resize((size.x * size.y) as usize, false);
    }
    pub fn is_in_bounds(&self, position: Vec2i) -> bool {
        position.x >= 0 && position.y >= 0 && position.x < self.size.x && position.y < self.size.y
    }
    pub fn get_cell(&self, position: Vec2i) -> Result<bool, MazeError> {
        if self.is_in_bounds(position) {
            Ok(self.cells[(self.size.x * position.y + position.x) as usize])
        } else {
            Err(MazeError::OutOfBounds)
        }
    }
    pub fn get_neighbor_cells_4_way(&self, position: Vec2i) -> Vec<(Vec2i, bool)> {
        self.get_neighbor_cells(position, &Vec2i::DIRECTIONS_4_WAY)
    }
    pub fn get_neighbor_cells_4_way_step(&self, position: Vec2i, step: i32) -> Vec<(Vec2i, bool)> {
        self.get_neighbor_cells(position, &Vec2i::DIRECTIONS_4_WAY.map(|x| x * step))
    }
    pub fn get_neighbor_cells(
        &self,
        position: Vec2i,
        neighbor_spots: &[Vec2i],
    ) -> Vec<(Vec2i, bool)> {
        let mut neighbor_cells = Vec::<(Vec2i, bool)>::new();
        for dir in neighbor_spots {
            let neighbor = position + *dir;
            if let Ok(neighbor_value) = self.get_cell(neighbor) {
                neighbor_cells.push((neighbor, neighbor_value));
            }
        }
        neighbor_cells
    }
    pub fn set_cell(&mut self, position: Vec2i, value: bool) -> Result<(), MazeError> {
        if self.is_in_bounds(position) {
            self.cells[(self.size.x * position.y + position.x) as usize] = value;
            Ok(())
        } else {
            Err(MazeError::OutOfBounds)
        }
    }
    pub fn fill(&mut self, value: bool) {
        self.cells.fill(value);
    }
}
impl Display for Map {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Maze({}, {}):\n", self.size.x, self.size.y)?;
        write!(f, "XX{}XX\n", "XX".repeat(self.size.x as usize))?;
        for y in 0..self.size.y {
            write!(f, "XX")?;
            for x in 0..self.size.x {
                if self.cells[(self.size.x * y + x) as usize] {
                    write!(f, "{}", "XX")?;
                } else {
                    write!(f, "  ")?;
                }
            }
            write!(f, "XX\n")?;
        }
        write!(f, "XX{}XX\n", "XX".repeat(self.size.x as usize))?;
        Ok(())
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MazeError {
    OutOfBounds,
}
// endregion

// region MazeGenerator
pub struct MazeGenerator {
    pub size: Vec2i,
}

impl MazeGenerator {
    pub fn new(size: Vec2i) -> MazeGenerator {
        MazeGenerator { size }
    }
    pub fn generate_random(&self, density: f32) -> Map {
        let mut rng = rand::thread_rng();
        let mut maze = Map::new(self.size);
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                maze.set_cell(Vec2i::new(x, y), rng.gen_bool(density.into()))
                    .unwrap();
            }
        }
        maze
    }
    pub fn generate_dfs_map_default(&self) -> Map {
        self.generate_dfs_map(Vec2i::ZERO)
    }
    pub fn generate_dfs_map(&self, start_pos: Vec2i) -> Map {
        // We only modify even # indicies
        // Slots each have a cell between them that can
        // either be a wall or an empty cell
        let mut rng = rand::thread_rng();
        let mut map = Map::new(self.size);
        map.fill(true);
        let mut visited_slots = HashSet::<Vec2i>::new();
        let mut current_slots_stack: Vec<(Vec2i, Option<Vec2i>)> = vec![(start_pos, None)];
        // Open up each slot
        while let Some((current_slot, prev_slot)) = current_slots_stack.pop() {
            if visited_slots.contains(&current_slot) {
                continue;
            }

            visited_slots.insert(current_slot);
            // Open up every slot we visit
            map.set_cell(current_slot, false).unwrap();

            // Open up the wall between current_cell and the neighbor
            if let Some(prev_slot) = prev_slot {
                let mid_cell = (current_slot + prev_slot) / 2;
                map.set_cell(mid_cell, false).unwrap();
            }

            // Add neighbors to stack
            let mut neighbors = map.get_neighbor_cells_4_way_step(current_slot, 2);
            if neighbors.len() > 0 {
                neighbors.shuffle(&mut rng);
                for (neighbor, _) in neighbors {
                    current_slots_stack.push((neighbor, Some(current_slot)));
                }
            }
        }
        map
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

pub struct Agent {
    /// Agent's mental map of the environment
    pub mind_map: Map,
    /// Current possition of the agent
    pub position: Vec2i,
    /// Goal position the agent wants to reach
    pub goal: Vec2i,
    /// Path to target.
    /// Stored in order from goal to target (reverse order), to promote fast popping
    pub target_path: Vec<Vec2i>,
    /// Status of the agent.
    /// - Inactive when the agent is first created
    /// - Running after the agent is initialized
    /// - Complete after the agent finishes path finding (Either the goal was reached or no path can be found)
    pub status: AgentStatus,
}
#[derive(Hash, Clone)]
struct AStarState {
    pub actual_cost: i32,
    pub heuristic_cost: i32,
    pub cell: Vec2i,
    pub parent_state: Option<Rc<AStarState>>,
}
impl AStarState {
    pub fn f_cost(&self) -> i32 {
        self.actual_cost + self.heuristic_cost
    }
    pub fn heap_f_cost(&self) -> i32 {
        // BinaryHeap is MaxHeap by default, so we need to negate the cost to make it a min heap
        -self.f_cost()
    }
}
impl Eq for AStarState {}
impl PartialEq for AStarState {
    fn eq(&self, other: &Self) -> bool {
        self.heap_f_cost() == other.heap_f_cost()
    }
}
impl PartialOrd for AStarState {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for AStarState {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.heap_f_cost().cmp(&other.heap_f_cost())
    }
}
impl Agent {
    pub fn new(map_size: Vec2i, start_pos: Vec2i, goal: Vec2i) -> Agent {
        Agent {
            position: start_pos,
            goal,
            target_path: Vec::new(),
            status: AgentStatus::Running,
            mind_map: Map::new(map_size),
        }
    }
    pub fn step(&mut self, world_map: &Map) {
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
                if let Some(next_pos) = self.target_path.last() {
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
            self.target_path = self.astar_pathfind();
        }
        if let Some(new_pos) = self.target_path.pop() {
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
    /// Attempts to find a path from the agent's current position to the goal
    ///
    /// Returns path from self to goal in reversed order (first = goal, last = path)
    pub fn astar_pathfind(&self) -> Vec<Vec2i> {
        let mut open_states = BinaryHeap::<Rc<AStarState>>::new();
        let mut closed_cells = HashSet::<Vec2i>::new();
        open_states.push(Rc::new(AStarState {
            actual_cost: 0,
            heuristic_cost: self.position.manhattan_distance(&self.goal),
            cell: self.position,
            parent_state: None,
        }));
        while let Some(curr_state) = open_states.pop() {
            closed_cells.insert(curr_state.cell);
            if curr_state.cell == self.goal {
                // Path has been found
                let mut path = vec![curr_state.cell];
                let mut curr_state_ref = &curr_state;
                // Create path by walking up the parent chain
                while let Some(parent_state) = curr_state_ref.parent_state.as_ref() {
                    path.push(parent_state.cell);
                    curr_state_ref = parent_state;
                }
                // No need to include the starting positioon
                path.pop();
                return path;
            }
            for (neighbor_cell, neighbor_filled) in
                self.mind_map.get_neighbor_cells_4_way(curr_state.cell)
            {
                // Only consider neighbors that are empty and are not explored yet
                if !neighbor_filled && !closed_cells.contains(&neighbor_cell) {
                    open_states.push(Rc::new(AStarState {
                        actual_cost: curr_state.actual_cost + 1, // Cost for every move is 1
                        heuristic_cost: neighbor_cell.manhattan_distance(&self.goal), // Heuristic is the manhattan distance between the cell and the goal
                        cell: neighbor_cell,
                        parent_state: Some(curr_state.clone()),
                    }));
                }
            }
        }
        // No path found
        vec![]
    }
}
// endregion

// region Simulation
pub struct Simulation {
    pub map: Map,
    pub agent: Agent,
    pub prev_agent_pos: Vec2i,
    pub steps: i32,
    pub result: Option<bool>,
}
impl Simulation {
    pub fn new(map: Map, agent: Agent) -> Simulation {
        Simulation {
            map,
            prev_agent_pos: agent.position,
            agent,
            steps: 0,
            result: None,
        }
    }
    pub fn is_running(&self) -> bool {
        self.result.is_none()
    }
    pub fn is_complete(&self) -> bool {
        self.result.is_some()
    }
    pub fn step(&mut self) {
        self.prev_agent_pos = self.agent.position;
        if self.is_complete() {
            return;
        }
        self.agent.step(&self.map);
        self.steps += 1;
        if let AgentStatus::Complete(result) = self.agent.status {
            self.result = Some(result);
        }
    }
    pub fn simulation_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "Simulation({}, {}) Step {}:\n",
            self.map.size.x, self.map.size.y, self.steps
        );
        for (line, line_2) in self.mind_map_str().lines().zip(self.full_map_str().lines()) {
            str += &format!("{}     {}\n", line, line_2);
        }
        str
    }
    pub fn full_map_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(self.map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        for y in 0..self.map.size.y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..self.map.size.x {
                let pos = Vec2i::new(x, y);
                if self.map.get_cell(pos).unwrap() {
                    str += &format!("{}", "XX".on_truecolor(125, 125, 125));
                } else {
                    if self.prev_agent_pos == pos {
                        str += &format!("{}", "AA".on_truecolor(69, 163, 65));
                    } else if self.agent.goal == pos {
                        str += &format!(
                            "{}",
                            "GG".truecolor(240, 227, 46).on_truecolor(230, 137, 39)
                        );
                    } else {
                        str += "  ";
                    }
                }
            }
            str += &format!("{}\n", "XX".on_truecolor(125, 125, 125));
        }
        str += &format!(
            "{}\n",
            "XX".repeat(self.map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        str
    }
    pub fn mind_map_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(self.map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        let agent_path_hashset: HashSet<Vec2i> =
            self.agent.target_path.clone().into_iter().collect();
        for y in 0..self.map.size.y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..self.map.size.x {
                let pos = Vec2i::new(x, y);
                if self.agent.mind_map.get_cell(pos).unwrap() {
                    str += &format!("{}", "XX".on_truecolor(125, 125, 125));
                } else {
                    if self.prev_agent_pos == pos {
                        str +=
                            &format!("{}", "AA".truecolor(172, 240, 46).on_truecolor(69, 163, 65));
                    } else if self.agent.goal == pos {
                        str += &format!(
                            "{}",
                            "GG".truecolor(240, 227, 46).on_truecolor(230, 137, 39)
                        );
                    } else if agent_path_hashset.contains(&pos) || pos == self.agent.position {
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
            "XX".repeat(self.map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        str
    }
}
impl Display for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.simulation_str())?;
        Ok(())
    }
}
// endregion

// region SimulationBuilder
pub struct SimulationBuilder;
impl SimulationBuilder {
    pub fn from_map(map: Map) -> Simulation {
        let agent = Agent::new(
            map.size,
            Vec2i::ZERO,
            Self::find_longest_reachable_goal(&map, Vec2i::ZERO),
        );
        Simulation::new(map, agent)
    }

    fn find_longest_reachable_goal(map: &Map, pos: Vec2i) -> Vec2i {
        Self::find_longest_reachable_goal_dfs(map, pos, &mut HashSet::<Vec2i>::new(), 0).0
    }
    fn find_longest_reachable_goal_dfs(
        map: &Map,
        pos: Vec2i,
        visited: &mut HashSet<Vec2i>,
        path_length: i32,
    ) -> (Vec2i, i32) {
        visited.insert(pos);
        let mut longest_path = None;
        for (neighbor, neighbor_filled) in map.get_neighbor_cells_4_way(pos) {
            if !neighbor_filled && !visited.contains(&neighbor) {
                let path =
                    Self::find_longest_reachable_goal_dfs(map, neighbor, visited, path_length + 1);
                if path.1 > longest_path.get_or_insert(path).1 {
                    longest_path = Some(path);
                }
            }
        }
        // Return the longest path, or the path we have so far up to this point
        longest_path.unwrap_or((pos, path_length))
    }
}
// endregion

// region SimulationRunner
pub struct SimulationRunner {
    pub simulation: Simulation,
    pub interval: f32,
    pub print: bool,
}
impl SimulationRunner {
    pub fn new(simulation: Simulation, interval: f32, print: bool) -> SimulationRunner {
        SimulationRunner {
            simulation,
            interval,
            print,
        }
    }
    pub fn run(&mut self) -> bool {
        if self.print {
            clearscreen::clear().unwrap();
            println!("🚀 Simulation Start");
            println!("{}", self.simulation);
        }
        while self.simulation.is_running() {
            self.simulation.step();
            clearscreen::clear().unwrap();
            println!("{}", self.simulation);
            if self.interval > 0.0 {
                sleep(Duration::from_secs_f32(self.interval))
            }
        }
        let result = self.simulation.result.unwrap();
        if self.print {
            clearscreen::clear().unwrap();
            self.simulation.step();
            println!("{}", self.simulation);
            println!("🛑 Simulation End. Goal Reached? {}", result);
        }
        result
    }
}
// endregion
