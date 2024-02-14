#![allow(dead_code)]
use owo_colors::OwoColorize;
use rand::{rngs::SmallRng, seq::SliceRandom, thread_rng, Rng, SeedableRng};
use std::{
    collections::{BinaryHeap, HashSet},
    env,
    fmt::{Debug, Display},
    hash::Hash,
    ops,
    rc::Rc,
    sync::Mutex,
    thread::sleep,
    time::{Duration, Instant},
    vec,
};

pub mod heap;

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut args: Vec<String> = env::args().collect();

    // Remove program name
    args.reverse();
    args.pop().unwrap();
    pub enum Mode {
        HeapTests,
        AutoTests,
        ManualTest,
    }
    let mut mode: Mode = Mode::ManualTest;
    if let Some(mode_str) = args.pop() {
        if mode_str.to_lowercase() == "heap" {
            mode = Mode::HeapTests;
        }
        if mode_str.to_lowercase() == "auto" {
            mode = Mode::AutoTests;
        } else if mode_str.to_lowercase() == "manual" {
            mode = Mode::ManualTest;
        }
    }

    match mode {
        Mode::HeapTests => heap::heap_tests(args),
        Mode::AutoTests => auto_tests(args),
        Mode::ManualTest => manual_test(args),
    }
}

fn auto_tests(mut args: Vec<String>) {
    let mut rng_seed = thread_rng().gen();
    if let Some(seed_str) = args.pop() {
        if let Ok(seed_val) = seed_str.parse() {
            rng_seed = seed_val;
        }
    }

    let trials = 50;
    let mut builder = MazeBuilder::new(rng_seed, Vec2i::new(51, 51));
    let mut maps = Vec::new();
    for _ in 0..trials {
        let map = builder.generate_dfs_map_default();
        maps.push(map);
    }

    auto_test_behavior(
        "A* Forward",
        rng_seed,
        &maps,
        AStarBehavior::new(true, BreakTieMode::HigherGCost),
    );

    auto_test_behavior(
        "A* Backward",
        rng_seed,
        &maps,
        AStarBehavior::new(false, BreakTieMode::HigherGCost),
    );

    auto_test_behavior(
        "A* Forward Lower G",
        rng_seed,
        &maps,
        AStarBehavior::new(true, BreakTieMode::LowerGCost),
    );

    auto_test_behavior(
        "A* Forward Higher G",
        rng_seed,
        &maps,
        AStarBehavior::new(true, BreakTieMode::HigherGCost),
    );

    auto_test_behavior(
        "Adaptive A*",
        rng_seed,
        &maps,
        AdaptiveAStarBehavior::new(BreakTieMode::HigherGCost),
    );
}

fn auto_test_behavior<T: AgentBehavior + Clone + 'static>(
    name: &str,
    rng_seed: u64,
    maps: &Vec<WallMap>,
    agent_behavior: T,
) {
    let mut total_time: f32 = 0.0;
    for map in maps.iter() {
        let simulation = SimulationBuilder::from_map(
            rng_seed,
            map.clone(),
            true,
            Box::new(agent_behavior.clone()),
        )
        .expect("Expect building to work.");
        let mut instant_runner = SimulationRunner::new_instant(simulation);
        let now = Instant::now();
        {
            instant_runner.run_instant();
        }
        let elapsed_time = now.elapsed();
        total_time += elapsed_time.as_millis() as f32;
    }
    let average_time = total_time / maps.len() as f32;
    println!("{}: Average: {} ms", name, average_time);
}

fn manual_test(mut args: Vec<String>) {
    let mut rng_seed = thread_rng().gen();
    if let Some(seed_str) = args.pop() {
        if let Ok(seed_val) = seed_str.parse() {
            rng_seed = seed_val;
        }
    }

    let mut size = Vec2i::new(25, 25);
    if let Some(x_str) = args.pop() {
        if let Ok(x) = x_str.parse() {
            if let Some(y_str) = args.pop() {
                if let Ok(y) = y_str.parse() {
                    size = Vec2i::new(x, y);
                }
            }
        }
    }
    let mut interval: f32 = 0.25;
    if let Some(interval_str) = args.pop() {
        if let Ok(interval_val) = interval_str.parse() {
            interval = interval_val;
        }
    }

    pub enum MapType {
        DFS,
        DFSRand,
    }
    let mut map_type = MapType::DFSRand;
    if let Some(map_type_str) = args.pop() {
        if map_type_str.to_lowercase() == "dfs" {
            map_type = MapType::DFS;
        } else if map_type_str.to_lowercase() == "dfs-rand" {
            map_type = MapType::DFSRand;
        }
    }

    let mut reachable_goal = false;
    if let Some(reachable_goal_str) = args.pop() {
        if reachable_goal_str.to_lowercase() == "reachable" {
            reachable_goal = true;
        } else if reachable_goal_str.to_lowercase() == "unreachable" {
            reachable_goal = false;
        }
    }

    let mut break_tie_mode = BreakTieMode::None;
    if let Some(break_tie_mode_str) = args.pop() {
        if break_tie_mode_str.to_lowercase() == "none" {
            break_tie_mode = BreakTieMode::None;
        } else if break_tie_mode_str.to_lowercase() == "high-g" {
            break_tie_mode = BreakTieMode::HigherGCost;
        } else if break_tie_mode_str.to_lowercase() == "low-g" {
            break_tie_mode = BreakTieMode::LowerGCost;
        }
    }

    let mut is_forward = false;
    if let Some(reachable_goal_str) = args.pop() {
        if reachable_goal_str.to_lowercase() == "forward" {
            is_forward = true;
        } else if reachable_goal_str.to_lowercase() == "backward" {
            is_forward = false;
        }
    }

    let mut ai_behavior: Box<dyn AgentBehavior> = Box::new(AStarBehavior {
        break_tie_mode,
        is_forward,
    });
    if let Some(ai_behavior_str) = args.pop() {
        if ai_behavior_str.to_lowercase() == "astar" {
            ai_behavior = Box::new(AStarBehavior::new(is_forward, break_tie_mode));
        } else if ai_behavior_str.to_lowercase() == "adaptive-astar" {
            ai_behavior = Box::new(AdaptiveAStarBehavior::new(break_tie_mode));
        }
    }

    let mut map_gen = MazeBuilder::new(rng_seed, size);
    let rand_map = match map_type {
        MapType::DFS => map_gen.generate_dfs_map_default(),
        MapType::DFSRand => map_gen.generate_dfs_map_random_default(),
    };
    let simulation =
        SimulationBuilder::from_map(rng_seed, rand_map.clone(), reachable_goal, ai_behavior);
    if let Some(simulation) = simulation {
        let mut runner = SimulationRunner::new(simulation, interval, true);
        runner.run();
    } else {
        println!("⛔ All map cells are connected, cannot set unreachable goal.");
        println!("{}", rand_map);
    }
}

// region Vec2i
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Hash)]
pub struct Vec2i {
    pub x: i32,
    pub y: i32,
}
impl Vec2i {
    pub const ZERO: Vec2i = Vec2i { x: 0, y: 0 };
    pub const ONE: Vec2i = Vec2i { x: 1, y: 1 };
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
pub type WallMap = Map<bool>;
impl WallMap {
    pub fn find_unreachable_goal(&self, rng_seed: u64, pos: Vec2i) -> Option<Vec2i> {
        let mut rng = SmallRng::seed_from_u64(rng_seed);
        let mut visited = HashSet::<Vec2i>::new();
        let mut connected_components = vec![pos];
        self.dfs_visit(pos, &mut visited);
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                let pos = Vec2i::new(x, y);
                if self.get_cell(pos).unwrap() || visited.contains(&pos) {
                    continue;
                }
                self.dfs_visit(pos, &mut visited);
                connected_components.push(pos);
            }
        }
        if connected_components.len() == 1 {
            None
        } else {
            Some(connected_components[rng.gen_range(1..connected_components.len())])
        }
    }
    pub fn dfs_visit(&self, pos: Vec2i, visited: &mut HashSet<Vec2i>) {
        visited.insert(pos);
        for (neighbor, neighbor_filled) in self.get_neighbor_cells_4_way(pos) {
            if !neighbor_filled && !visited.contains(&neighbor) {
                self.dfs_visit(neighbor, visited);
            }
        }
    }

    pub fn find_longest_reachable_goal(&self, pos: Vec2i) -> Vec2i {
        self.find_longest_reachable_goal_dfs(pos, &mut HashSet::<Vec2i>::new(), 0)
            .0
    }
    fn find_longest_reachable_goal_dfs(
        &self,
        pos: Vec2i,
        visited: &mut HashSet<Vec2i>,
        path_length: i32,
    ) -> (Vec2i, i32) {
        visited.insert(pos);
        let mut longest_path = None;
        for (neighbor, neighbor_filled) in self.get_neighbor_cells_4_way(pos) {
            if !neighbor_filled && !visited.contains(&neighbor) {
                let path = self.find_longest_reachable_goal_dfs(neighbor, visited, path_length + 1);
                if path.1 > longest_path.get_or_insert(path).1 {
                    longest_path = Some(path);
                }
            }
        }
        // Return the longest path, or the path we have so far up to this point
        longest_path.unwrap_or((pos, path_length))
    }
}
impl Display for WallMap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WallMap({}, {}):\n", self.size.x, self.size.y)?;
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

pub type ConnectedComponentsMap = Map<u32>;
impl ConnectedComponentsMap {
    pub fn from_wall_map(map: &WallMap) -> ConnectedComponentsMap {
        let mut connected_components_map = ConnectedComponentsMap::new(map.size);
        let mut visited = HashSet::<Vec2i>::new();
        let mut component_id = 1;
        for y in 0..map.size.y {
            for x in 0..map.size.x {
                let pos = Vec2i::new(x, y);
                if visited.contains(&pos) {
                    continue;
                }
                connected_components_map.wall_map_dfs_visit(&map, pos, &mut visited, component_id);
                component_id += 1;
            }
        }
        connected_components_map
    }
    fn wall_map_dfs_visit(
        &mut self,
        wall_map: &WallMap,
        pos: Vec2i,
        visited: &mut HashSet<Vec2i>,
        component_id: u32,
    ) {
        visited.insert(pos);
        self.set_cell(pos, component_id).unwrap();
        for (neighbor, neighbor_filled) in wall_map.get_neighbor_cells_4_way(pos) {
            if !neighbor_filled && !visited.contains(&neighbor) {
                self.wall_map_dfs_visit(wall_map, neighbor, visited, component_id);
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Map<T: Clone + Default> {
    cells: Vec<T>,
    size: Vec2i,
}

impl<T: Clone + Default> Map<T> {
    pub fn new<TNew>(size: Vec2i) -> Map<TNew>
    where
        TNew: Default + Clone,
    {
        Map::<TNew> {
            cells: vec![TNew::default(); (size.x * size.y) as usize],
            size,
        }
    }
    pub fn size(&self) -> &Vec2i {
        &self.size
    }
    pub fn cells(&self) -> &Vec<T> {
        &self.cells
    }
    pub fn resize(&mut self, size: Vec2i) {
        self.cells.resize((size.x * size.y) as usize, T::default());
        self.size = size;
    }
    pub fn is_in_bounds(&self, position: Vec2i) -> bool {
        position.x >= 0 && position.y >= 0 && position.x < self.size.x && position.y < self.size.y
    }
    pub fn get_cell(&self, position: Vec2i) -> Result<T, MapError> {
        if self.is_in_bounds(position) {
            Ok(self.cells[(self.size.x * position.y + position.x) as usize].clone())
        } else {
            Err(MapError::OutOfBounds)
        }
    }
    pub fn get_neighbor_cells_4_way(&self, position: Vec2i) -> Vec<(Vec2i, T)> {
        self.get_neighbor_cells(position, &Vec2i::DIRECTIONS_4_WAY)
    }
    pub fn get_neighbor_cells_4_way_step(&self, position: Vec2i, step: i32) -> Vec<(Vec2i, T)> {
        self.get_neighbor_cells(position, &Vec2i::DIRECTIONS_4_WAY.map(|x| x * step))
    }
    pub fn get_neighbor_cells(&self, position: Vec2i, neighbor_spots: &[Vec2i]) -> Vec<(Vec2i, T)> {
        let mut neighbor_cells = Vec::<(Vec2i, T)>::new();
        for dir in neighbor_spots {
            let neighbor = position + *dir;
            if let Ok(neighbor_value) = self.get_cell(neighbor) {
                neighbor_cells.push((neighbor, neighbor_value));
            }
        }
        neighbor_cells
    }
    pub fn set_cell(&mut self, position: Vec2i, value: T) -> Result<(), MapError> {
        if self.is_in_bounds(position) {
            self.cells[(self.size.x * position.y + position.x) as usize] = value;
            Ok(())
        } else {
            Err(MapError::OutOfBounds)
        }
    }
    pub fn fill(&mut self, value: T) {
        self.cells.fill(value);
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum MapError {
    OutOfBounds,
}
// endregion

// region MazeBuilder
pub struct MazeBuilder {
    pub size: Vec2i,
    pub rng: SmallRng,
}
impl MazeBuilder {
    pub fn new(rng_seed: u64, size: Vec2i) -> MazeBuilder {
        MazeBuilder {
            size,
            rng: SmallRng::seed_from_u64(rng_seed),
        }
    }
    pub fn generate_random(&mut self, density: f32) -> WallMap {
        let mut maze = WallMap::new(self.size);
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                maze.set_cell(Vec2i::new(x, y), self.rng.gen_bool(density.into()))
                    .unwrap();
            }
        }
        maze
    }
    /// Get the starting (top left most) cell that
    /// is aligned with a slot pattern that includes pos
    fn get_start_slot_cell(pos: Vec2i) -> Vec2i {
        let mut start_cell = Vec2i::ZERO;
        if pos.x % 2 == 1 {
            start_cell.x = 1;
        }
        if pos.y % 2 == 1 {
            start_cell.y = 1;
        }
        start_cell
    }
    pub fn generate_dfs_map_random_default(&mut self) -> WallMap {
        self.generate_dfs_map_random(Vec2i::ZERO)
    }
    pub fn generate_dfs_map_random(&mut self, start_pos: Vec2i) -> WallMap {
        let mut map = self.generate_dfs_map(start_pos);
        let total_cells = (self.size.x * self.size.y) as usize;

        self.set_rand_cells(&mut map, start_pos, total_cells / 14, false);
        self.set_rand_cells(&mut map, start_pos, total_cells / 18, true);
        map
    }
    fn set_rand_cells(&mut self, map: &mut WallMap, start_pos: Vec2i, amount: usize, value: bool) {
        let start_cell = Self::get_start_slot_cell(start_pos);
        let half_size = self.size / 2;
        for _ in 0..amount {
            let rand_slot = Vec2i::new(
                self.rng.gen_range(0..half_size.x),
                self.rng.gen_range(0..half_size.y),
            );
            let rand_pos = start_cell + rand_slot * 2;
            let rand_dir = Vec2i::DIRECTIONS_4_WAY[self.rng.gen_range(0..4)];
            let rand_wall_pos = rand_pos + rand_dir;
            // Random cell might be out of bounds, but we don't care
            let _ = map.set_cell(rand_wall_pos, value);
        }
    }
    pub fn generate_dfs_map_default(&mut self) -> WallMap {
        self.generate_dfs_map(Vec2i::ZERO)
    }
    pub fn generate_dfs_map(&mut self, start_pos: Vec2i) -> WallMap {
        // We only modify even # indicies
        // Slots each have a cell between them that can
        // either be a wall or an empty cell
        let mut map = WallMap::new(self.size);
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
                neighbors.shuffle(&mut self.rng);
                for (neighbor, _) in neighbors {
                    current_slots_stack.push((neighbor, Some(current_slot)));
                }
            }
        }
        map
    }
}
// endregion

// region AgentBehavior
pub trait AgentBehavior: Debug {
    /// Initializes the behavior
    fn init(&mut self, _map_size: Vec2i) {}
    /// Attempts to find a path from the agent's current position to the goal
    ///
    /// Returns path from self to goal in reversed order (first = goal, last = path)
    fn pathfind(&mut self, agent: &Agent) -> Vec<Vec2i>;
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
        start: Vec2i,
        end: Vec2i,
        break_tie_mode: BreakTieMode,
    ) -> Vec<Vec2i> {
        let mut open_states = BinaryHeap::<Rc<AStarState>>::new();
        let mut closed_cells = HashSet::<Vec2i>::new();

        open_states.push(Rc::new(AStarState::from_g_h_costs(
            0,
            start.manhattan_distance(&end),
            start,
            None,
            break_tie_mode,
        )));
        while let Some(curr_state) = open_states.pop() {
            if closed_cells.contains(&curr_state.cell) {
                // There was an earlier path that beat us to the punch.
                // This state is no longer relevant.
                continue;
            }
            closed_cells.insert(curr_state.cell);
            if curr_state.cell == end {
                // Path has been found
                let mut path = vec![curr_state.cell];
                let mut curr_state_ref = &curr_state;
                // Create path by walking up the parent chain
                while let Some(parent_state) = curr_state_ref.parent_state.as_ref() {
                    path.push(parent_state.cell);
                    curr_state_ref = parent_state;
                }
                return path;
            }
            for (neighbor_cell, neighbor_filled) in
                wall_map.get_neighbor_cells_4_way(curr_state.cell)
            {
                // Only consider neighbors that are empty and are not explored yet
                if !neighbor_filled && !closed_cells.contains(&neighbor_cell) {
                    open_states.push(Rc::new(AStarState::from_g_h_costs(
                        curr_state.actual_cost + 1,             // Cost for every move is 1
                        neighbor_cell.manhattan_distance(&end), // Heuristic is the manhattan distance between the cell and the goal
                        neighbor_cell,
                        Some(curr_state.clone()),
                        break_tie_mode,
                    )));
                }
            }
        }
        // No path found
        vec![]
    }
}
impl AgentBehavior for AStarBehavior {
    fn pathfind(&mut self, agent: &Agent) -> Vec<Vec2i> {
        let mut start = agent.position;
        let mut end = agent.goal;
        if !self.is_forward {
            start = agent.goal;
            end = agent.position;
        }
        let mut path = Self::astar_pathfind(&agent.mind_map, start, end, self.break_tie_mode);
        if !self.is_forward {
            path.reverse()
        }
        // No need to include the starting positioon
        path.pop();
        path
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveAStarBehavior {
    pub break_tie_mode: BreakTieMode,
    pub actual_cost_map: Map<i32>,
}
impl AdaptiveAStarBehavior {
    pub fn new(break_tie_mode: BreakTieMode) -> AdaptiveAStarBehavior {
        AdaptiveAStarBehavior {
            break_tie_mode,
            actual_cost_map: Map::<i32>::new(Vec2i::ONE),
        }
    }
    fn calc_heuristic_cost(&self, agent: &Agent, position: Vec2i) -> i32 {
        let cached_actual_cost = self.actual_cost_map.get_cell(position).unwrap();
        if cached_actual_cost > 0 {
            // h(s) = g(s_goal) - g(s)
            self.actual_goal_cost(agent) - cached_actual_cost
        } else {
            position.manhattan_distance(&agent.goal)
        }
    }
    fn actual_goal_cost(&self, agent: &Agent) -> i32 {
        self.actual_cost_map.get_cell(agent.goal).unwrap()
    }
}
impl AgentBehavior for AdaptiveAStarBehavior {
    fn init(&mut self, map_size: Vec2i) {
        self.actual_cost_map.resize(map_size);
        self.actual_cost_map.fill(-1);
    }
    fn pathfind(&mut self, agent: &Agent) -> Vec<Vec2i> {
        let mut open_states = BinaryHeap::<Rc<AStarState>>::new();
        let mut closed_cells = HashSet::<Vec2i>::new();

        self.actual_cost_map.set_cell(agent.position, 0).unwrap();
        open_states.push(Rc::new(AStarState::from_g_h_costs(
            0,
            self.calc_heuristic_cost(agent, agent.position),
            agent.position,
            None,
            self.break_tie_mode,
        )));
        while let Some(curr_state) = open_states.pop() {
            if closed_cells.contains(&curr_state.cell) {
                // There was an earlier path that beat us to the punch.
                // This state is no longer relevant.
                continue;
            }
            closed_cells.insert(curr_state.cell);
            if curr_state.cell == agent.goal {
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
                agent.mind_map.get_neighbor_cells_4_way(curr_state.cell)
            {
                // Only consider neighbors that are empty and are not explored yet
                if !neighbor_filled && !closed_cells.contains(&neighbor_cell) {
                    let actual_cost = curr_state.actual_cost + 1;
                    self.actual_cost_map
                        .set_cell(agent.position, actual_cost)
                        .unwrap();
                    open_states.push(Rc::new(AStarState::from_g_h_costs(
                        actual_cost, // Cost for every move is 1
                        self.calc_heuristic_cost(agent, neighbor_cell),
                        neighbor_cell,
                        Some(curr_state.clone()),
                        self.break_tie_mode,
                    )));
                }
            }
        }
        // No path found
        vec![]
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
    pub target_path: Vec<Vec2i>,
    /// Status of the agent.
    /// - Inactive when the agent is first created
    /// - Running after the agent is initialized
    /// - Complete after the agent finishes path finding (Either the goal was reached or no path can be found)
    pub status: AgentStatus,
    /// Behavior of the agent
    pub behavior: Mutex<Box<dyn AgentBehavior>>,
}
#[derive(Clone)]
struct AStarState {
    pub heap_cost: i32,
    pub actual_cost: i32,
    pub heuristic_cost: i32,
    pub cell: Vec2i,
    pub parent_state: Option<Rc<AStarState>>,
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
        }
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
        self.heap_cost.cmp(&other.heap_cost)
    }
}
impl Agent {
    pub fn new(start_pos: Vec2i, goal: Vec2i, behavior: Box<dyn AgentBehavior>) -> Agent {
        Agent {
            position: start_pos,
            goal,
            target_path: Vec::new(),
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
            self.target_path = self.behavior.try_lock().unwrap().pathfind(self);
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
}
// endregion

// region Simulation
#[derive(Debug)]
pub struct Simulation {
    pub wall_map: WallMap,
    pub connected_components_map: ConnectedComponentsMap,
    pub agent: Agent,
    pub prev_agent_pos: Vec2i,
    pub steps: i32,
    pub result: Option<bool>,
}
impl Simulation {
    pub fn new(map: WallMap, mut agent: Agent) -> Simulation {
        let connected_components_map = ConnectedComponentsMap::from_wall_map(&map);
        agent.init(map.size);
        Simulation {
            wall_map: map,
            connected_components_map,
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
        self.agent.step(&self.wall_map);
        self.steps += 1;
        if let AgentStatus::Complete(result) = self.agent.status {
            self.result = Some(result);
        }
    }
    pub fn simulation_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "Simulation({}, {}) Step {}:\n",
            self.wall_map.size.x, self.wall_map.size.y, self.steps
        );
        for (line, line_2) in self.mind_map_str().lines().zip(self.full_map_str().lines()) {
            str += &format!("{}     {}\n", line, line_2);
        }
        str
    }
    fn get_connected_component_string(&self, pos: Vec2i) -> String {
        if let Ok(comp_id) = self.connected_components_map.get_cell(pos) {
            match comp_id % 7 {
                0 => format!("{}", " 0".truecolor(168, 41, 32)),
                1 => format!("{}", " 1".truecolor(168, 89, 32)),
                2 => format!("{}", " 2".truecolor(168, 161, 32)),
                3 => format!("{}", " 3".truecolor(105, 168, 32)),
                4 => format!("{}", " 4".truecolor(32, 168, 154)),
                5 => format!("{}", " 5".truecolor(114, 32, 168)),
                6 => format!("{}", " 6".truecolor(168, 32, 123)),
                _ => "  ".to_owned(),
            }
        } else {
            "  ".to_owned()
        }
    }
    pub fn full_map_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(self.wall_map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        for y in 0..self.wall_map.size.y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..self.wall_map.size.x {
                let pos = Vec2i::new(x, y);
                if self.wall_map.get_cell(pos).unwrap() {
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
                        str += &self.get_connected_component_string(pos);
                    }
                }
            }
            str += &format!("{}\n", "XX".on_truecolor(125, 125, 125));
        }
        str += &format!(
            "{}\n",
            "XX".repeat(self.wall_map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        str
    }
    pub fn mind_map_str(&self) -> String {
        let mut str = String::new();
        str += &format!(
            "{}\n",
            "XX".repeat(self.wall_map.size.x as usize + 2)
                .on_truecolor(125, 125, 125)
        );
        let agent_path_hashset: HashSet<Vec2i> =
            self.agent.target_path.clone().into_iter().collect();
        for y in 0..self.wall_map.size.y {
            str += &format!("{}", "XX".on_truecolor(125, 125, 125));
            for x in 0..self.wall_map.size.x {
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
            "XX".repeat(self.wall_map.size.x as usize + 2)
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
    pub fn from_map(
        rng_seed: u64,
        map: WallMap,
        set_reachable_goal: bool,
        agent_behavior: Box<dyn AgentBehavior>,
    ) -> Option<Simulation> {
        let goal = {
            if set_reachable_goal {
                map.find_longest_reachable_goal(Vec2i::ZERO)
            } else {
                if let Some(unreachable_goal) = map.find_unreachable_goal(rng_seed, Vec2i::ZERO) {
                    unreachable_goal
                } else {
                    return None;
                }
            }
        };
        let agent = Agent::new(Vec2i::ZERO, goal, agent_behavior);
        Some(Simulation::new(map, agent))
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
    pub fn new_instant(simulation: Simulation) -> SimulationRunner {
        SimulationRunner::new(simulation, 0.0, false)
    }
    pub fn new(simulation: Simulation, interval: f32, print: bool) -> SimulationRunner {
        SimulationRunner {
            simulation,
            interval,
            print,
        }
    }
    pub fn run(&mut self) -> bool {
        if self.is_instant() {
            return self.run_instant();
        }

        if self.print {
            clearscreen::clear().unwrap();
            println!("🚀 Simulation Start");
            println!("{}", self.simulation);
        }
        while self.simulation.is_running() {
            self.simulation.step();
            if self.print {
                clearscreen::clear().unwrap();
                println!("{}", self.simulation);
            }
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
    pub fn run_instant(&mut self) -> bool {
        while self.simulation.is_running() {
            self.simulation.step();
        }
        self.simulation.result.unwrap()
    }
    pub fn is_instant(&self) -> bool {
        !self.print && self.interval <= 0.0
    }
}
// endregion
