use std::{env, error::Error, fs::File, io::Read, iter::zip, os::raw};

use ndarray::{prelude::*, ShapeError};
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};

type Label = i32;
type RawData = Vec2D<f32>;
type Features = ArrayD<f32>;
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

struct LabelDataFeatureParser {
    pub data_parser: DataParser,
    pub feature_extractor: Box<dyn FeatureExtractor>,
    pub label_parser: LabelParser,
}

#[derive(Debug)]
enum LabelDataFeatureParserError {
    DataParserError(std::io::Error),
    LabelParserError(Box<dyn Error>),
    FeatureExtractorError(ShapeError),
    MismatchedSize(usize, usize),
}

impl LabelDataFeatureParser {
    fn new<T: FeatureExtractor + 'static>(
        data_parser: DataParser,
        feature_extractor: T,
        label_parser: LabelParser,
    ) -> Self {
        LabelDataFeatureParser {
            data_parser,
            feature_extractor: Box::new(feature_extractor),
            label_parser,
        }
    }

    fn parse_files(
        &self,
        data_file_path: &str,
        label_file_path: &str,
    ) -> Result<Vec<(Features, Label)>, LabelDataFeatureParserError> {
        let raw_data = self
            .data_parser
            .parse_file(data_file_path)
            .map_err(|e| LabelDataFeatureParserError::DataParserError(e))?;
        let labels = self
            .label_parser
            .parse_file(label_file_path)
            .map_err(|e| LabelDataFeatureParserError::LabelParserError(e))?;
        if raw_data.len() != labels.len() {
            return Err(LabelDataFeatureParserError::MismatchedSize(
                raw_data.len(),
                labels.len(),
            ));
        }
        let mut features_list = Vec::new();
        for data in raw_data {
            let features = self
                .feature_extractor
                .extract_features(data)
                .map_err(|e| LabelDataFeatureParserError::FeatureExtractorError(e))?;
            features_list.push(features);
        }
        Ok(zip(features_list, labels).collect())
    }
}

trait FeatureExtractor {
    fn extract_features(&self, data: RawData) -> Result<Features, ShapeError>;
}

struct PixelFeatureExtractor;
impl FeatureExtractor for PixelFeatureExtractor {
    fn extract_features(&self, data: RawData) -> Result<Features, ShapeError> {
        Ok(ArrayD::from_shape_vec(
            IxDyn(&[data.width, data.height]),
            data.array,
        )?)
    }
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
    let mut seed: u64 = 1234;
    if let Some(value) = args.get(2) {
        if let Ok(value) = value.parse::<u64>() {
            seed = value;
        }
    }
    let mut epochs: usize = 3;
    if let Some(value) = args.get(3) {
        if let Ok(value) = value.parse::<usize>() {
            epochs = value;
        }
    }
    let mut learn_rate: f32 = 0.01;
    if let Some(value) = args.get(4) {
        if let Ok(value) = value.parse::<f32>() {
            learn_rate = value;
        }
    }
    let split_chars = DataParser::NEW_LINE_CHARS.to_vec();
    let char_range = vec![' ', '+', '#'];
    let label_data_parser = LabelDataFeatureParser::new(
        DataParser::new(width, height, split_chars, char_range),
        PixelFeatureExtractor,
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
    let mut rng = StdRng::seed_from_u64(seed);

    // Neural Network Structure:
    //
    // Input (width * height) -> Hidden (20) -> Output (10 digits)

    // Weights for input to hidden layer
    let w_i_h = Array::random_using((20, width * height), Uniform::new(-0.5, 0.5), &mut rng);
    let b_i_h = Array::<f64, _>::zeros((20, 1));

    // Weights for hidden layer to output layer
    let w_h_o = Array::random_using((10, 20), Uniform::new(-0.5, 0.5), &mut rng);
    let b_i_o = Array::<f64, _>::zeros((10, 1));

    let mut number_correct: usize = 0;
    for epoch in 0..epochs {
        for (features, label) in labelled_training_data.iter() {
            // TODO
        }
    }
}

fn train_faces(args: &[String]) {}
