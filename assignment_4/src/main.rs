use std::{
    env,
    error::Error,
    fs::File,
    io::{stdin, Read},
};

use ndarray::prelude::*;
use ndarray_rand::{
    rand::{rngs::StdRng, SeedableRng},
    rand_distr::{num_traits::ToPrimitive, Uniform},
    RandomExt,
};

type RawData = Vec2D<f64>;
type Label = usize;
type Features = Vec<f64>;
type LabelVec = Array2<f64>;
type FeaturesVec = Array2<f64>;
type LabelFeatureDataSet = Vec<(RawData, Features, Label)>;
type LabelFeatureVecDataSet = Vec<(RawData, FeaturesVec, LabelVec)>;

#[derive(Debug, Clone)]
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

    fn parse_file(&self, file_path: &str) -> Result<Vec<Label>, Box<dyn Error>> {
        let mut file = File::open(file_path)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let mut labels = Vec::new();
        for label in buffer.split_whitespace() {
            labels.push(label.parse::<Label>()?)
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
                    // println!("finished parsing one sample:");
                    // print_data(&curr_sample);
                    samples_vec.push(curr_sample);
                    y_pos = 0;
                    curr_sample = RawData::new(self.width, self.height);
                }
            }
            if let Some(index) = self.value_char_range.iter().position(|&x| x == char) {
                let percent = index as f64 / (self.value_char_range.len() - 1) as f64;
                curr_sample.set(x_pos, y_pos, percent);
            }
            x_pos += 1;
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
    FeatureExtractorError(Box<dyn Error>),
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
    ) -> Result<LabelFeatureDataSet, LabelDataFeatureParserError> {
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
        for data in raw_data.iter() {
            let features = self
                .feature_extractor
                .extract_features(data.clone())
                .map_err(|e| LabelDataFeatureParserError::FeatureExtractorError(e))?;
            features_list.push(features);
        }
        Ok(raw_data
            .into_iter()
            .zip(features_list)
            .zip(labels)
            .map(|((x, y), z)| (x, y, z))
            .collect())
    }

    fn parse_files_to_vec(
        &self,
        data_file_path: &str,
        label_file_path: &str,
        label_range: usize,
    ) -> Result<LabelFeatureVecDataSet, LabelDataFeatureParserError> {
        let data_set = self.parse_files(data_file_path, label_file_path)?;
        Ok(self.convert_to_vec(data_set, label_range))
    }

    fn convert_to_vec(
        &self,
        data: Vec<(RawData, Features, Label)>,
        label_range: usize,
    ) -> LabelFeatureVecDataSet {
        data.into_iter()
            .map(|(data, features, label)| {
                let feat_vec: FeaturesVec = Array2::from_shape_vec((features.len(), 1), features)
                    .expect("Expect conversion to workfrom_shape");
                let feat_label: LabelVec =
                    Array2::from_shape_fn(
                        (label_range, 1),
                        |(i, j)| {
                            if i == label {
                                1.0
                            } else {
                                0.0
                            }
                        },
                    );
                (data, feat_vec, feat_label)
            })
            .collect()
    }
}

trait FeatureExtractor {
    fn extract_features(&self, data: RawData) -> Result<Features, Box<dyn Error>>;
}

struct PixelFeatureExtractor;
impl FeatureExtractor for PixelFeatureExtractor {
    fn extract_features(&self, data: RawData) -> Result<Features, Box<dyn Error>> {
        // Get a (# pixels) x 1 vector
        Ok(data.array)
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
    let mut learn_rate: f64 = 0.01;
    if let Some(value) = args.get(4) {
        if let Ok(value) = value.parse::<f64>() {
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
        .parse_files_to_vec(
            "data/digitdata/trainingimages",
            "data/digitdata/traininglabels",
            10,
        )
        .expect("Expect label and data files to be parsable");
    let labelled_test_data = label_data_parser
        .parse_files_to_vec("data/digitdata/testimages", "data/digitdata/testlabels", 10)
        .expect("Expect label and data files to be parsable");
    println!("1️⃣  Training digits");
    println!("   Training set size: {}", labelled_training_data.len());
    println!("   Test set size: {}", labelled_test_data.len());
    let mut rng = StdRng::seed_from_u64(seed);

    // Neural Network Structure:
    //
    // Input (width * height) -> Hidden (20) -> Output (10 digits)

    // Weights + bias for input to hidden layer
    let mut w_i_h =
        Array2::<f64>::random_using((20, width * height), Uniform::new(-0.5, 0.5), &mut rng);
    let mut b_i_h = Array2::<f64>::zeros((20, 1));

    // Weights + bias for hidden layer to output layer
    let mut w_h_o = Array2::<f64>::random_using((10, 20), Uniform::new(-0.5, 0.5), &mut rng);
    let mut b_h_o = Array2::<f64>::zeros((10, 1));

    let mut number_correct: usize;
    for epoch in 0..epochs {
        number_correct = 0;
        for (_data, features, label) in labelled_training_data.iter() {
            // Forward propagation
            let h_pre = &b_i_h + &w_i_h.dot(features);
            let h = activation_func(h_pre);

            let o_pre = &b_h_o + &w_h_o.dot(&h);
            let o = activation_func(o_pre);

            // Cost function
            // let _e = (1.0 / o.len() as f64) * (&o - label).mapv_into(|v| v.powi(2)).sum();

            if label_certainty_from_vec(&o).0 == label_certainty_from_vec(label).0 {
                number_correct += 1;
            }

            // Backpropagation
            let delta_o = &o - label;
            w_h_o = w_h_o - learn_rate * &delta_o.dot(&h.t());
            b_h_o = b_h_o - learn_rate * &delta_o;

            let delta_h = (w_h_o.t().dot(&delta_o)) * activation_func_deriv(h);
            w_i_h = w_i_h - learn_rate * &delta_h.dot(&features.t());
            b_i_h = b_i_h - learn_rate * &delta_h;
        }

        println!(
            "Epoch {}:\n  Accuracy: {:.2}%",
            epoch,
            (number_correct as f64 / labelled_training_data.len() as f64 * 100.0)
        )
    }

    loop {
        println!(
            "Enter a test image number (0 - {}) to test: ('q' to quit)",
            labelled_test_data.len()
        );
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Expected input");
        if let Ok(index) = input.trim().parse::<usize>() {
            let (data, features, label) = &labelled_test_data[index];

            println!("Data ({}):", label_certainty_from_vec(label).0);
            print_data(data);

            // Forward propagation
            let h_pre = &b_i_h + &w_i_h.dot(features);
            let h = activation_func(h_pre);

            let o_pre = &b_h_o + &w_h_o.dot(&h);
            let o = activation_func(o_pre);

            let (label, certainty) = label_certainty_from_vec(&o);

            println!(
                "🧠 Neural network predicts: {} with {:.2}% certainty\n",
                label,
                certainty * 100.0
            )
        } else if input.trim() == "q" {
            break;
        }
    }
}

fn print_data(data: &RawData) {
    for y in 0..data.height {
        for x in 0..data.width {
            let val = *data.get(x, y);
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

fn label_certainty_from_vec(array: &Array2<f64>) -> (usize, f64) {
    let mut max = 0.0;
    let mut max_index = 0;
    let mut index = 0;
    for elem in array {
        if *elem > max {
            max = *elem;
            max_index = index;
        }
        index += 1;
    }
    return (max_index, max);
}

fn activation_func(amount: Array2<f64>) -> Array2<f64> {
    amount.mapv_into(|v| 1.0 / (1.0 + (-v).exp()))
}

fn activation_func_deriv(amount: Array2<f64>) -> Array2<f64> {
    amount.mapv_into(|v| (v * (1.0 - v)))
}

fn train_faces(args: &[String]) {}
