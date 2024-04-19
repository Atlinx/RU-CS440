use std::{env, error::Error, fs::File, io::Read, iter::zip};

type Label = i32;
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

struct LabelParser {}

impl LabelParser {
    fn new() -> Self {
        LabelParser {}
    }

    fn parse_file(&self, file_path: &str) -> Result<Vec<i32>, Box<dyn Error>> {
        let mut file = File::open(file_path)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let mut labels = Vec::new();
        for label in buffer.split_whitespace() {
            labels.push(label.parse::<i32>()?)
        }
        Ok(labels)
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
    fn parse_file(&self, file_path: &str) -> Result<Vec<RawData>, std::io::Error> {
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

struct LabelDataParser {
    pub data_parser: DataParser,
    pub label_parser: LabelParser,
}

#[derive(Debug)]
enum LabelDataParserError {
    DataParserError(std::io::Error),
    LabelParserError(Box<dyn Error>),
    MismatchedSize(usize, usize),
}

impl LabelDataParser {
    fn new(data_parser: DataParser, label_parser: LabelParser) -> Self {
        LabelDataParser {
            data_parser,
            label_parser,
        }
    }

    fn parse_files(
        &self,
        data_file_path: &str,
        label_file_path: &str,
    ) -> Result<Vec<(RawData, Label)>, LabelDataParserError> {
        let raw_data = self
            .data_parser
            .parse_file(data_file_path)
            .map_err(|e| LabelDataParserError::DataParserError(e))?;
        let labels = self
            .label_parser
            .parse_file(label_file_path)
            .map_err(|e| LabelDataParserError::LabelParserError(e))?;
        if raw_data.len() != labels.len() {
            return Err(LabelDataParserError::MismatchedSize(
                raw_data.len(),
                labels.len(),
            ));
        }
        Ok(zip(raw_data, labels).collect())
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
    let width: usize = 28;
    let height = 28;
    if let Some(data_set) = args.get(1) {
        if data_set == "test" {}
    }
    let split_chars = DataParser::NEW_LINE_CHARS.to_vec();
    let char_range = vec![' ', '+', '#'];
    let label_data_parser = LabelDataParser::new(
        DataParser::new(width, height, split_chars, char_range),
        LabelParser::new(),
    );
    let labelled_training_data = label_data_parser
        .parse_files(
            "data/digitdata/trainingimages",
            "data/digitdata/traininglabels",
        )
        .expect("Expect label and data files to be parsable");
    let labelled_test_data = label_data_parser
        .parse_files("data/digitdata/testimages", "data/digitdata/testlabels")
        .expect("Expect label and data files to be parsable");
    println!("1️⃣  Training digits");
    println!("   Training set size: {}", labelled_training_data.len());
    println!("   Test set size: {}", labelled_test_data.len());
    for (raw_data, label) in labelled_training_data {}
}

fn train_faces(args: &[String]) {}
