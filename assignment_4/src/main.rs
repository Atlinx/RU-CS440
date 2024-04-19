use std::{env, fs::File, io::Read};

type RawData = Vec2D<f32>;
type Features = Vec2D<f32>;
struct TrainingData {
    features: Features,
    label: u8,
}
type TrainingDataSet = Vec<TrainingData>;

#[derive(Debug)]
struct Vec2D<T: Default + Clone> {
    width: usize,
    height: usize,
    array: Vec<T>,
}

impl<T: Default + Clone> Vec2D<T> {
    fn new(width: usize, height: usize) -> Self {
        Vec2D {
            width,
            height,
            array: vec![T::default(); width * height],
        }
    }

    fn get(&self, x: usize, y: usize) -> &T {
        &self.array[y * self.width + x]
    }

    fn set(&mut self, x: usize, y: usize, value: T) {
        self.array[y * self.width + x] = value
    }
}

struct DataParser {
    width: usize,
    height: usize,
    split_chars: Vec<char>,
    value_char_range: Vec<char>,
}

impl DataParser {
    const NEW_LINE_CHARS: [char; 2] = ['\n', '\r'];

    fn new(
        width: usize,
        height: usize,
        split_chars: Vec<char>,
        value_char_range: Vec<char>,
    ) -> Self {
        DataParser {
            width,
            height,
            split_chars,
            value_char_range,
        }
    }
    fn parse_file(&self, file_path: String) -> Result<Vec<RawData>, std::io::Error> {
        let mut file = File::open(file_path)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let mut samples_vec = Vec::new();
        let mut curr_sample = RawData::new(self.width, self.height);
        let mut x_pos: usize = 0;
        let mut y_pos: usize = 0;
        let mut prev_char_was_split: bool = false;
        for char in buffer.chars() {
            if self.split_chars.contains(&char) {
                prev_char_was_split = true;
                continue;
            } else if prev_char_was_split {
                // We are at a new line
                prev_char_was_split = false;
                y_pos += 1;
                x_pos = 0;
                if y_pos >= self.height {
                    // We've finished parsing one sample
                    samples_vec.push(curr_sample);
                    y_pos = 0;
                    curr_sample = RawData::new(self.width, self.height);
                }
            }
            if let Some(index) = self.value_char_range.iter().position(|&x| x == char) {
                let percent = index as f32 / (self.value_char_range.len() - 1) as f32;
                curr_sample.set(x_pos, y_pos, percent);
            }
        }
        samples_vec.push(curr_sample);
        Ok(samples_vec)
    }
}

trait FeatureExtractor {
    fn extract_features(data: RawData) -> Features;
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut args: Vec<String> = env::args().collect();
    if let Some(value) = args.get(1).cloned() {
        // Remove program name and digits/faces args
        args.remove(0);
        args.remove(0);
        if value == "digits" {
            train_digits(&args);
        } else {
            train_faces(&args);
        }
    } else {
        train_digits(&args);
    }
}

fn train_digits(args: &[String]) {
    let mut width: usize = 28;
    if let Some(value) = args.get(1) {
        width = value.parse().expect("Expected width to be of usize.");
    }
    let mut height = 70;
    if let Some(value) = args.get(2) {
        height = value.parse().expect("Expected height to be of usize.");
    }
    let mut file_path = "data/digitdata/trainingimages".to_owned();
    if let Some(value) = args.get(3) {
        file_path = value.to_owned();
    }
    let mut split_chars = DataParser::NEW_LINE_CHARS.to_vec();
    if let Some(value) = args.get(4) {
        split_chars = value.chars().collect();
    }
    let mut char_range = vec![' ', '+', '#'];
    if let Some(value) = args.get(5) {
        char_range = value.chars().collect();
    }
    let parser = DataParser::new(width, height, split_chars, char_range);
    let raw_data = parser
        .parse_file(file_path)
        .expect("Expect file to be parsable.");
}

fn train_faces(args: &[String]) {
    // TODO
}
