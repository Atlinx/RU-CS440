use std::fmt::Display;

fn main() {
    println!("Hello, world!");
    println!("printing a vec: {}", Vec2::new(5, 10))
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
struct Vec2 {
    pub x: i32,
    pub y: i32,
}

impl Vec2 {
    pub fn new(x: i32, y: i32) -> Vec2 {
        Vec2 { x, y }
    }
}

impl Display for Vec2 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vec2({}, {})", self.x, self.y)
    }
}

#[derive(Debug)]
struct Maze {
    cells: Vec<bool>,
    size: Vec2,
}

impl Maze {
    fn new(size: Vec2) -> Maze {
        Maze {
            cells: vec![false; (size.x * size.y) as usize],
            size,
        }
    }
    fn size(&self) -> &Vec2 {
        &self.size
    }
    fn cells(&self) -> &Vec<bool> {
        &self.cells
    }
    fn resize(&mut self, size: Vec2) {
        self.cells.resize((size.x * size.y) as usize, false);
    }
    fn is_in_bounds(&self, position: Vec2) -> bool {
        position.x >= 0 && position.y >= 0 && position.x < self.size.x && position.y < self.size.y
    }
    fn get_cell(&self, position: Vec2) -> Result<bool, MazeError> {
        if self.is_in_bounds(position) {
            Ok(self.cells[(self.size.x * position.y + position.x) as usize])
        } else {
            Err(MazeError::OutOfBounds)
        }
    }
    fn set_cell(&mut self, position: Vec2, value: bool) -> Result<(), MazeError> {
        if self.is_in_bounds(position) {
            self.cells[(self.size.x * position.y + position.x) as usize] = value;
            Ok(())
        } else {
            Err(MazeError::OutOfBounds)
        }
    }
    fn state_string(&self) -> String {
        let mut str = String::new();
        for y in 0..self.size.y {
            for x in 0..self.size.x {
                if self.cells[(self.size.x * y + x) as usize] {
                    str += " "
                } else {
                    str += "X"
                }
            }
            str += "\n"
        }
        str
    }
}

enum MazeError {
    OutOfBounds,
}
