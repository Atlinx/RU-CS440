use std::{fmt::Display, ops};

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
    pub const DIRECTIONS_4_WAY: [Vec2i; 4] = [Vec2i::UP, Vec2i::LEFT, Vec2i::DOWN, Vec2i::RIGHT];

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

// region Utils
pub trait StringExten {
    fn join_sides_spaced(&self, other: &Self) -> Self;
}
impl StringExten for String {
    fn join_sides_spaced(&self, other: &Self) -> Self {
        let mut str = String::new();
        for (line, line_2) in self.lines().zip(other.lines()) {
            str += &format!("{}     {}\n", line, line_2);
        }
        str
    }
}
// endregion
