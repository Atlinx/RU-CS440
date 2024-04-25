use std::{
    env,
    error::Error,
    fs::File,
    io::{stdin, Read},
    vec,
};

use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray_rand::{
    rand::{rngs::StdRng, Rng, SeedableRng},
    rand_distr::Uniform,
    RandomExt,
};

type RawData = Vec2D<f64>;
type Label = usize;
type Features = Vec<f64>;
type LabelVec = Array2<f64>;
type FeaturesVec = Array2<f64>;
type LabelFeatureDataSet = Vec<(RawData, Features, Label)>;
type LabelFeatureVecDataSet = Vec<(RawData, FeaturesVec, LabelVec)>;
type LabelFeatureVecDataSetSlice = [(RawData, FeaturesVec, LabelVec)];

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

struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    layers: Vec<usize>,
    activation_func: Box<dyn Fn(f64) -> f64 + Sync + Send>,
    activation_func_deriv: Box<dyn Fn(f64) -> f64 + Sync + Send>,
    learn_rate_func: Box<dyn Fn(f64) -> f64 + Sync + Send>,
    learn_rate: f64,
}

impl NeuralNetwork {
    fn from_structure<
        R: Rng,
        F: Fn(f64) -> f64 + 'static + Sync + Send,
        FD: Fn(f64) -> f64 + 'static + Sync + Send,
        FL: Fn(f64) -> f64 + 'static + Sync + Send,
    >(
        mut rng: &mut R,
        layers: &[usize],
        activation_func: F,
        activation_func_deriv: FD,
        learn_rate_func: FL,
    ) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..(layers.len() - 1) {
            let weight = Array2::<f64>::random_using(
                (layers[i + 1], layers[i]),
                Uniform::new(-0.5, 0.5),
                &mut rng,
            );
            let bias = Array2::<f64>::zeros((layers[i + 1], 1));
            weights.push(weight);
            biases.push(bias);
        }
        NeuralNetwork {
            weights,
            biases,
            layers: layers.iter().copied().collect(),
            activation_func: Box::new(activation_func),
            activation_func_deriv: Box::new(activation_func_deriv),
            learn_rate: learn_rate_func(0.0),
            learn_rate_func: Box::new(learn_rate_func),
        }
    }

    // Returns number of layers in the network
    fn layers(&self) -> usize {
        self.layers.len()
    }

    // Returns number of nodes in the network
    fn nodes(&self) -> usize {
        let mut total = 0;
        for nodes in self.layers.iter() {
            total += nodes;
        }
        total
    }

    // Trains the neural network on an entire data set
    fn train_data_set(
        &mut self,
        data_set: &LabelFeatureVecDataSetSlice,
        epochs: usize,
        print: bool,
    ) {
        if print {
            println!(
                "✏️  Start training on dataset (size: {}, epochs: {})",
                data_set.len(),
                epochs
            )
        }
        for epoch in 0..epochs {
            let mut number_correct: usize = 0;
            self.learn_rate = (self.learn_rate_func)(epoch as f64);
            for (_, features, label) in data_set {
                let correct = self.train_data_point(features, label);
                if correct {
                    number_correct += 1;
                }
            }
            if print {
                println!(
                    "   📆 Epoch {}: Accuracy: {:.2}% Learn rate: {}",
                    epoch,
                    (number_correct as f64 / data_set.len() as f64 * 100.0).round(),
                    self.learn_rate
                )
            }
        }
        if print {
            println!("🏁 Finished training")
        }
    }

    // Trains the neural network on a single data point.
    // Returns the whether the neural network correctly predicted
    // the data point or not.
    fn train_data_point(&mut self, features: &FeaturesVec, label: &LabelVec) -> bool {
        // Forward propagation
        let mut a = features.clone();
        let mut layer_values = vec![a.clone()];
        for i in 0..self.weights.len() {
            let bias = &self.biases[i];
            let weights = &self.weights[i];
            a = bias + weights.dot(&a);
            a.mapv_inplace(&self.activation_func);
            layer_values.push(a.clone());
        }

        let output = layer_values.last().expect("Expect last value to exist");
        let correct = label_certainty_from_vec(output).0 == label_certainty_from_vec(label).0;
        // Back propagation
        let mut delta = output - label;
        for i in (0..self.weights.len()).rev() {
            let res = self.learn_rate * &delta.dot(&layer_values[i].t());
            self.weights[i] = &self.weights[i] - res;
            self.biases[i] = &self.biases[i] - self.learn_rate * &delta;
            layer_values[i].mapv_inplace(&self.activation_func_deriv);
            delta = (self.weights[i].t().dot(&delta)) * &layer_values[i];
        }

        correct
    }

    // Tests the neural network on a data point.
    // Returns the a tuple containing the prediction label and certainty of the prediction.
    fn test_data_point(&self, features: &FeaturesVec) -> (usize, f64) {
        let mut a = features.clone();
        for i in 0..self.weights.len() {
            let bias = &self.biases[i];
            let weights = &self.weights[i];
            let a_pre = bias + weights.dot(&a);
            a = a_pre.mapv_into(&self.activation_func);
        }
        label_certainty_from_vec(&a)
    }

    // Returns the accuracy of the neural network accross an entire data set.
    fn test_accuracy(&self, data_set: &LabelFeatureVecDataSetSlice) -> f64 {
        let mut number_correct: usize = 0;
        for (_, features, label) in data_set {
            let (predicted_label_index, _) = self.test_data_point(features);
            let correct = predicted_label_index == label_certainty_from_vec(label).0;
            if correct {
                number_correct += 1;
            }
        }
        number_correct as f64 / data_set.len() as f64
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
    let mut seed: u64 = 1234;
    if let Some(value) = args.get(0) {
        if let Ok(value) = value.parse::<u64>() {
            seed = value;
        }
    }
    let mut training_data_percent: f64 = 1.0;
    if let Some(value) = args.get(1) {
        if let Ok(value) = value.parse::<f64>() {
            training_data_percent = value / 100.0;
        }
    }
    let mut intermediate_layers = vec![20];
    if let Some(value) = args.get(2) {
        intermediate_layers = value
            .split(" ")
            .filter_map(|x| x.parse::<usize>().ok())
            .collect();
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

    let mut layers = Vec::new();
    layers.push(width * height);
    layers.append(&mut intermediate_layers);
    layers.push(10);
    println!("1️⃣  Training digits");
    println!("   Training set size: {}", labelled_training_data.len());
    println!("   Test set size: {}", labelled_test_data.len());
    println!(
        "🔨 Settings\n   RNG seed: {}\n   Used training data: {:.2}%\n   Layers: {:?}\n   Epochs: {}\n   Learn rate: {}",
        seed, training_data_percent * 100.0, layers, epochs, learn_rate
    );
    let mut rng = StdRng::seed_from_u64(seed);
    let mut neural_network = NeuralNetwork::from_structure(
        &mut rng,
        &layers,
        activation_func,
        activation_func_deriv,
        move |x| learn_rate,
    );
    let used_training_data_size =
        (labelled_training_data.len() as f64 * training_data_percent).round() as usize;
    let used_training_data = &labelled_training_data[..used_training_data_size];
    neural_network.train_data_set(used_training_data, epochs, true);
    let accuracy = neural_network.test_accuracy(&labelled_test_data);
    println!("📄 Neural network accuracy: {:.2}%", accuracy * 100.0);
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

            let (label, certainty) = neural_network.test_data_point(features);

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

fn activation_func(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn activation_func_deriv(x: f64) -> f64 {
    x * (1.0 - x)
}

fn train_faces(args: &[String]) {}
