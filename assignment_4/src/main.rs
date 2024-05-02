use std::{env, io::stdin, vec};

use ai::AI;
use ndarray_rand::rand::{rngs::StdRng, SeedableRng};

use crate::{
    ai::{
        activation_func, activation_func_deriv, label_certainty_from_vec,
        neural_network::NeuralNetwork, perceptron::Perceptron,
    },
    feature_extractor::pixel_feature_extractor::PixelFeatureExtractor,
    parser::{
        data_parser::DataParser, label_data_feature_parser::LabelDataFeatureParser,
        label_parser::LabelParser,
    },
    types::RawDataExtens,
};

pub mod ai;
pub mod feature_extractor;
pub mod parser;
pub mod types;

struct AIConfig {
    name: String,
    ai: Box<dyn AI>,
    epochs: usize,
}

struct TrainingValidationDatasetConfig {
    title: String,
    width: usize,
    height: usize,
    split_chars: Vec<char>,
    char_range: Vec<char>,
    training_data_path: String,
    training_labels_path: String,
    validation_data_path: String,
    validation_labels_path: String,
    training_data_percent: f64,
    label_range: usize,
    label_to_text_fn: Box<dyn Fn(usize) -> String + Sync + Send>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TrainingDatasetType {
    Digits,
    Faces,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AiType {
    NeuralNetwork,
    Perceptron,
}

fn train_and_test_ai(mut ai_config: AIConfig, dataset_config: TrainingValidationDatasetConfig) {
    let label_data_parser = LabelDataFeatureParser::new(
        DataParser::new(
            dataset_config.width,
            dataset_config.height,
            dataset_config.split_chars,
            dataset_config.char_range,
        ),
        PixelFeatureExtractor,
        LabelParser::new(),
    );
    let labelled_training_data = label_data_parser
        .parse_files_to_vec(
            &dataset_config.training_data_path,
            &dataset_config.training_labels_path,
            dataset_config.label_range,
        )
        .expect("Expect label and data files to be parsable");
    let labelled_test_data = label_data_parser
        .parse_files_to_vec(
            &dataset_config.validation_data_path,
            &dataset_config.validation_labels_path,
            dataset_config.label_range,
        )
        .expect("Expect label and data files to be parsable");

    println!("{}", dataset_config.title);
    println!("   Training set size: {}", labelled_training_data.len());
    println!("   Test set size: {}", labelled_test_data.len());
    println!(
        "   Used training data: {:.2}%",
        dataset_config.training_data_percent * 100.0
    );
    let used_training_data_size = (labelled_training_data.len() as f64
        * dataset_config.training_data_percent)
        .round() as usize;
    let used_training_data = &labelled_training_data[..used_training_data_size];
    ai_config
        .ai
        .train_data_set(used_training_data, ai_config.epochs, true);
    let accuracy = ai_config.ai.test_accuracy(&labelled_test_data);
    println!("📄 {} accuracy: {:.2}%", ai_config.name, accuracy * 100.0);
    loop {
        println!(
            "Enter a test image number (0 - {}) to test: ('q' to quit)",
            labelled_test_data.len()
        );
        let mut input = String::new();
        stdin().read_line(&mut input).expect("Expected input");
        if let Ok(index) = input.trim().parse::<usize>() {
            let (data, features, label) = &labelled_test_data[index];

            let label_index = label_certainty_from_vec(label).0;
            println!(
                "Data ({}: {}):",
                (dataset_config.label_to_text_fn)(label_index),
                label_index,
            );
            data.print();

            let (guess_label_index, guess_certainty) = ai_config.ai.test_data_point(features);

            println!(
                "🧠 {} predicts: ({}: {}) with {:.2}% certainty\n",
                ai_config.name,
                (dataset_config.label_to_text_fn)(guess_label_index),
                guess_label_index,
                guess_certainty * 100.0
            )
        } else if input.trim() == "q" {
            break;
        }
    }
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut args: Vec<String> = env::args().collect();
    args.reverse();
    // Remove the name of the program
    args.pop();

    let mut dataset_type = TrainingDatasetType::Digits;
    if let Some(value) = args.pop() {
        dataset_type = match value.as_str() {
            "faces" => TrainingDatasetType::Faces,
            _ => TrainingDatasetType::Digits,
        }
    }

    let mut ai_type = AiType::NeuralNetwork;
    if let Some(value) = args.pop() {
        ai_type = match value.as_str() {
            "perceptron" => AiType::Perceptron,
            _ => AiType::NeuralNetwork,
        }
    }

    let mut seed: u64 = 1234;
    if let Some(value) = args.pop() {
        if let Ok(value) = value.parse::<u64>() {
            seed = value;
        }
    }

    let mut training_data_percent: f64 = 1.0;
    if let Some(value) = args.pop() {
        if let Ok(value) = value.parse::<f64>() {
            training_data_percent = value / 100.0;
        }
    }

    let mut intermediate_layers = vec![20];
    if ai_type == AiType::NeuralNetwork {
        if let Some(value) = args.pop() {
            intermediate_layers = value
                .split(" ")
                .filter_map(|x| x.parse::<usize>().ok())
                .collect();
        }
    }

    let mut epochs: usize = 3;
    if let Some(value) = args.pop() {
        if let Ok(value) = value.parse::<usize>() {
            epochs = value;
        }
    }

    let mut learn_rate: f64 = 0.01;
    if let Some(value) = args.pop() {
        if let Ok(value) = value.parse::<f64>() {
            learn_rate = value;
        }
    }

    let dataset_config = match dataset_type {
        TrainingDatasetType::Digits => TrainingValidationDatasetConfig {
            title: "1️⃣  Digits".to_owned(),
            width: 28,
            height: 28,
            split_chars: DataParser::NEW_LINE_CHARS.to_vec(),
            char_range: vec![' ', '+', '#'],
            training_data_path: "data/digitdata/trainingimages".to_owned(),
            training_labels_path: "data/digitdata/traininglabels".to_owned(),
            validation_data_path: "data/digitdata/testimages".to_owned(),
            validation_labels_path: "data/digitdata/testlabels".to_owned(),
            training_data_percent,
            label_range: 10,
            label_to_text_fn: Box::new(|label| label.to_string()),
        },
        TrainingDatasetType::Faces => TrainingValidationDatasetConfig {
            title: "😀 Faces".to_owned(),
            width: 60,
            height: 70,
            split_chars: DataParser::NEW_LINE_CHARS.to_vec(),
            char_range: vec![' ', '#'],
            training_data_path: "data/facedata/facedatatrain".to_owned(),
            training_labels_path: "data/facedata/facedatatrainlabels".to_owned(),
            validation_data_path: "data/facedata/facedatatest".to_owned(),
            validation_labels_path: "data/facedata/facedatatestlabels".to_owned(),
            training_data_percent,
            label_range: 2,
            label_to_text_fn: Box::new(|label| {
                match label {
                    1 => "Face",
                    _ => "Not Face",
                }
                .to_owned()
            }),
        },
    };

    let ai_name: String;
    let ai: Box<dyn AI>;

    match ai_type {
        AiType::NeuralNetwork => {
            let mut layers = Vec::new();
            layers.push(dataset_config.width * dataset_config.height);
            layers.append(&mut intermediate_layers);
            layers.push(dataset_config.label_range);
            println!("💡 Neural Network");
            println!(
                "🔨 Settings\n   RNG seed: {}\n   Used training data: {:.2}%\n   Layers: {:?}\n   Epochs: {}\n   Learn rate: {}",
                seed, training_data_percent * 100.0, layers, epochs, learn_rate
            );
            let mut rng = StdRng::seed_from_u64(seed);

            ai_name = "Neural network".to_owned();
            ai = Box::new(NeuralNetwork::from_structure(
                &mut rng,
                &layers,
                activation_func,
                activation_func_deriv,
                move |_| learn_rate,
            ))
        }
        AiType::Perceptron => {
            println!("👀 Perceptron");
            println!(
                "🔨 Settings\n   Epochs: {}\n   Learn rate: {}",
                epochs, learn_rate
            );

            ai_name = "Perceptron".to_owned();
            ai = Box::new(Perceptron::new(learn_rate))
        }
    };

    let ai_config = AIConfig {
        ai,
        epochs,
        name: ai_name,
    };

    train_and_test_ai(ai_config, dataset_config);
}
