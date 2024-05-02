use ndarray::Array2;

pub type RawData = Vec2D<f64>;
pub type Label = usize;
pub type Features = Vec<f64>;
pub type LabelVec = Array2<f64>;
pub type FeaturesVec = Array2<f64>;
pub type LabelFeatureDataSet = Vec<(RawData, Features, Label)>;
pub type LabelFeatureVecDataSet = Vec<(RawData, FeaturesVec, LabelVec)>;
pub type LabelFeatureVecDataSetSlice = [(RawData, FeaturesVec, LabelVec)];

#[derive(Debug, Clone)]
pub struct Vec2D<T: Default + Clone> {
    pub width: usize,
    pub height: usize,
    pub array: Vec<T>,
}

impl<T: Default + Clone> Vec2D<T> {
    pub fn new(width: usize, height: usize) -> Self {
        Vec2D {
            width,
            height,
            array: vec![T::default(); width * height],
        }
    }

    pub fn get(&self, x: usize, y: usize) -> &T {
        &self.array[y * self.width + x]
    }

    pub fn set(&mut self, x: usize, y: usize, value: T) {
        self.array[y * self.width + x] = value
    }
}

pub trait RawDataExtens {
    fn print(&self);
}

impl RawDataExtens for RawData {
    fn print(&self) {
        for y in 0..self.height {
            for x in 0..self.width {
                let val = *self.get(x, y);
                if val > 0.5 {
                    print!("#")
                } else if val > 0.0 {
                    print!("+")
                } else {
                    print!(" ")
                }
            }
            println!()
        }
    }
}
