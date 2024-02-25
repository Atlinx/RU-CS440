use rand::prelude::*;
use std::{
    collections::{HashSet, VecDeque},
    fmt::Display,
};

use crate::prelude::*;

// region MapType
pub enum MapType {
    DFS,
    DFSRand,
    Rand1,
    Rand2,
    Rand3,
    Rand4,
    Rand5,
    Rand6,
    Rand7,
    Rand8,
    Rand9,
}
impl MapType {
    pub fn generate_map(&self, map_gen: &mut MazeBuilder) -> WallMap {
        match self {
            MapType::DFS => map_gen.generate_dfs_map_default(),
            MapType::DFSRand => map_gen.generate_dfs_map_random_default(),
            MapType::Rand1 => map_gen.generate_random(0.1),
            MapType::Rand2 => map_gen.generate_random(0.2),
            MapType::Rand3 => map_gen.generate_random(0.3),
            MapType::Rand4 => map_gen.generate_random(0.4),
            MapType::Rand5 => map_gen.generate_random(0.5),
            MapType::Rand6 => map_gen.generate_random(0.7),
            MapType::Rand7 => map_gen.generate_random(0.7),
            MapType::Rand8 => map_gen.generate_random(0.8),
            MapType::Rand9 => map_gen.generate_random(0.9),
        }
    }
    pub fn parse_map_type_str(map_type_str: String) -> MapType {
        if map_type_str.to_lowercase() == "dfs" {
            MapType::DFS
        } else if map_type_str.to_lowercase() == "dfs-rand" {
            MapType::DFSRand
        } else if map_type_str.to_lowercase() == "rand-1" {
            MapType::Rand1
        } else if map_type_str.to_lowercase() == "rand-2" {
            MapType::Rand2
        } else if map_type_str.to_lowercase() == "rand-3" {
            MapType::Rand3
        } else if map_type_str.to_lowercase() == "rand-4" {
            MapType::Rand4
        } else if map_type_str.to_lowercase() == "rand-5" {
            MapType::Rand5
        } else if map_type_str.to_lowercase() == "rand-6" {
            MapType::Rand6
        } else if map_type_str.to_lowercase() == "rand-7" {
            MapType::Rand7
        } else if map_type_str.to_lowercase() == "rand-8" {
            MapType::Rand8
        } else if map_type_str.to_lowercase() == "rand-9" {
            MapType::Rand9
        } else {
            MapType::DFSRand
        }
    }
}
impl Display for MapType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                MapType::DFS => "DFS",
                MapType::DFSRand => "DFS Rand",
                MapType::Rand1 => "Rand 1",
                MapType::Rand2 => "Rand 2",
                MapType::Rand3 => "Rand 3",
                MapType::Rand4 => "Rand 4",
                MapType::Rand5 => "Rand 5",
                MapType::Rand6 => "Rand 6",
                MapType::Rand7 => "Rand 7",
                MapType::Rand8 => "Rand 8",
                MapType::Rand9 => "Rand 9",
            }
        )
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
        if visited.contains(&pos) {
            return;
        }
        let mut frontier_queue = VecDeque::<Vec2i>::new();
        frontier_queue.push_back(pos);
        visited.insert(pos);
        while let Some(pos) = frontier_queue.pop_front() {
            for (neighbor, neighbor_filled) in self.get_neighbor_cells_4_way(pos) {
                if !neighbor_filled && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    frontier_queue.push_back(neighbor);
                }
            }
        }
    }
    pub fn find_longest_reachable_goal(&self, pos: Vec2i) -> Vec2i {
        let mut frontier_queue = VecDeque::<(Vec2i, i32)>::new();
        frontier_queue.push_back((pos, 0));
        let mut visited = HashSet::<Vec2i>::new();
        visited.insert(pos);
        let mut longest_path = None;
        while let Some((pos, path_length)) = frontier_queue.pop_front() {
            if path_length > longest_path.get_or_insert((pos, path_length)).1 {
                longest_path = Some((pos, path_length));
            }
            for (neighbor, neighbor_filled) in self.get_neighbor_cells_4_way(pos) {
                if !neighbor_filled && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    frontier_queue.push_back((neighbor, path_length + 1))
                }
            }
        }
        longest_path.unwrap().0
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
        if visited.contains(&pos) {
            return;
        }
        let mut frontier_queue = VecDeque::<Vec2i>::new();
        frontier_queue.push_back(pos);
        visited.insert(pos);
        while let Some(pos) = frontier_queue.pop_front() {
            self.set_cell(pos, component_id).unwrap();
            for (neighbor, neighbor_filled) in wall_map.get_neighbor_cells_4_way(pos) {
                if !neighbor_filled && !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    frontier_queue.push_back(neighbor);
                }
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
        maze.set_cell(Vec2i::ZERO, false).unwrap();
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
