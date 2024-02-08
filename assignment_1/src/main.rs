#![allow(dead_code)]
use colored::Colorize;
use rand::{seq::SliceRandom, thread_rng, Rng};
use std::{borrow::BorrowMut, collections::HashSet, fmt::Display, ops};

fn main() {
    let map_gen = MazeGenerator::new(Vec2i::new(51, 51));
    let mut rand_map = map_gen.generate_random(0.5);
    println!("0.5 Density Map: \n{}", rand_map);
    rand_map = map_gen.generate_random(0.7);
    println!("0.7 Density Map: \n{}", rand_map);
    rand_map = map_gen.generate_dfs_map_default();
    println!("Rand DFS Map: \n{}", rand_map);
}

// region Utils
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
    fn new(size: Vec2i) -> Map {
        Map {
            cells: vec![false; (size.x * size.y) as usize],
            size,
        }
    }
    fn size(&self) -> &Vec2i {
        &self.size
    }
    fn cells(&self) -> &Vec<bool> {
        &self.cells
    }
    fn resize(&mut self, size: Vec2i) {
        self.cells.resize((size.x * size.y) as usize, false);
    }
    fn is_in_bounds(&self, position: Vec2i) -> bool {
        position.x >= 0 && position.y >= 0 && position.x < self.size.x && position.y < self.size.y
    }
    fn get_cell(&self, position: Vec2i) -> Result<bool, MazeError> {
        if self.is_in_bounds(position) {
            Ok(self.cells[(self.size.x * position.y + position.x) as usize])
        } else {
            Err(MazeError::OutOfBounds)
        }
    }
    fn set_cell(&mut self, position: Vec2i, value: bool) -> Result<(), MazeError> {
        if self.is_in_bounds(position) {
            self.cells[(self.size.x * position.y + position.x) as usize] = value;
            Ok(())
        } else {
            Err(MazeError::OutOfBounds)
        }
    }
    fn fill(&mut self, value: bool) {
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
enum MazeError {
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
        let mut maze = Map::new(self.size);
        maze.fill(true);
        let mut visited_slots = HashSet::<Vec2i>::new();
        let mut current_slots_stack: Vec<(Vec2i, Option<Vec2i>)> = vec![(start_pos, None)];
        // Open up each slot
        while let Some((current_slot, prev_slot)) = current_slots_stack.pop() {
            if visited_slots.contains(&current_slot) {
                continue;
            }

            visited_slots.insert(current_slot);
            // Open up every slot we visit
            maze.set_cell(current_slot, false).unwrap();

            // Open up the wall between current_cell and the neighbor
            if let Some(prev_slot) = prev_slot {
                let mid_cell = (current_slot + prev_slot) / 2;
                maze.set_cell(mid_cell, false).unwrap();
            }

            // Add neighbors to stack
            let mut neighbors = self.get_slot_neighbors(current_slot);
            if neighbors.len() > 0 {
                neighbors.shuffle(&mut rng);
                for neighbor in neighbors {
                    current_slots_stack.push((neighbor, Some(current_slot)));
                }
            }
        }
        maze
    }
    fn is_in_bounds(&self, pos: Vec2i) -> bool {
        pos.x >= 0 && pos.y >= 0 && pos.x < self.size.x && pos.y < self.size.y
    }
    fn get_slot_neighbors(&self, pos: Vec2i) -> Vec<Vec2i> {
        self.get_cell_neighbors(pos, 2)
    }
    fn get_cell_neighbors(&self, pos: Vec2i, step: i32) -> Vec<Vec2i> {
        let mut neighbors = Vec::new();
        for dir in Vec2i::DIRECTIONS_4_WAY {
            let neighbor = pos + dir * step;
            if self.is_in_bounds(neighbor) {
                neighbors.push(neighbor)
            }
        }
        neighbors
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
    pub mind_map: Option<Map>,
    pub position: Vec2i,
    pub target_path: Vec<Vec2i>,
    pub status: AgentStatus,
}
impl Agent {
    pub fn new(start_pos: Vec2i) -> Agent {
        Agent {
            position: start_pos,
            target_path: Vec::new(),
            status: AgentStatus::Inactive,
            mind_map: None,
        }
    }
    pub fn init(&mut self, map_size: Vec2i) {
        self.mind_map = Some(Map::new(map_size));
        self.status = AgentStatus::Running
    }
    pub fn step(&mut self, world_map: &Map) {
        let map = self
            .mind_map
            .as_mut()
            .expect("Expect Agent to be initialized.");
        for dir in Vec2i::DIRECTIONS_4_WAY {
            let neighbor_pos = self.position + dir;
            if let Ok(neighbor_value) = world_map.get_cell(neighbor_pos) {
                map.set_cell(neighbor_pos, neighbor_value).unwrap();
            }
        }
    }
}
// endregion

// region Simulation
pub struct Simulation {
    pub map: Map,
    pub agent: Agent,
    pub steps: i32,
    pub result: Option<bool>,
}
impl Simulation {
    pub fn new(map: Map, mut agent: Agent) -> Simulation {
        agent.init(map.size);
        Simulation {
            map,
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
        if self.is_complete() {
            return;
        }
        self.agent.step(&self.map);
        self.steps += 1;
        if let AgentStatus::Complete(result) = self.agent.status {
            self.result = Some(result);
        }
    }
}
impl Display for Simulation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Simulation({}, {}) Step {}:\n",
            self.map.size.x, self.map.size.y, self.steps
        )?;
        write!(f, "XX{}XX\n", "XX".repeat(self.map.size.x as usize))?;
        for y in 0..self.map.size.y {
            write!(f, "XX")?;
            for x in 0..self.map.size.x {
                let pos = Vec2i::new(x, y);
                if self.map.get_cell(pos).unwrap() {
                    write!(f, "{}", "XX")?;
                } else {
                    if self.agent.position == pos {
                        write!(f, "{}", "AA".green())?;
                    } else if self.agent.target_path.contains(&pos) {
                        write!(f, "{}", "pp".cyan())?;
                    } else {
                        write!(f, "  ")?;
                    }
                }
            }
            write!(f, "XX\n")?;
        }
        write!(f, "XX{}XX\n", "XX".repeat(self.map.size.x as usize))?;
        Ok(())
    }
}
// endregion
