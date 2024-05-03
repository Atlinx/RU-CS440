use std::{env, vec};

use ai::AI;
use runner::{
    manual_runner::ManualRunner, sampler_runner::SamplerRunner,
    sampler_set_runner::SamplerSetRunner, AIConfig, CompleteDatasetConfig, Runner,
};

use crate::{
    ai::{
        activation_func, activation_func_deriv, neural_network::NeuralNetwork,
        perceptron::Perceptron,
    },
    parser::data_parser::DataParser,
};

pub mod ai;
pub mod feature_extractor;
pub mod parser;
pub mod runner;
pub mod types;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RunnerType {
    Manual,
    Sampler,
    SamplerSet,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DatasetType {
    Digits,
    Faces,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum AiType {
    NeuralNetwork,
    Perceptron,
}

fn main() {
    env::set_var("RUST_BACKTRACE", "1");
    let mut args: Vec<String> = env::args().collect();
    args.reverse();
    // Remove the name of the program
    args.pop();

    let mut runner_type = RunnerType::Manual;
    if let Some(value) = args.pop() {
        runner_type = match value.as_str() {
            "sampler" => RunnerType::Sampler,
            "sampler_set" => RunnerType::SamplerSet,
            _ => RunnerType::Manual,
        }
    }

    let mut sample_count: usize = 0;
    if runner_type == RunnerType::Sampler || runner_type == RunnerType::SamplerSet {
        if let Some(value) = args.pop() {
            if let Ok(value) = value.parse::<usize>() {
                sample_count = value;
            }
        }
    }

    let mut dataset_type = DatasetType::Digits;
    if let Some(value) = args.pop() {
        dataset_type = match value.as_str() {
            "faces" => DatasetType::Faces,
            _ => DatasetType::Digits,
        }
    }

    let mut training_data_percent: f64 = 1.0;
    if runner_type == RunnerType::Manual || runner_type == RunnerType::Sampler {
        if let Some(value) = args.pop() {
            println!("rec train data percent: {}", value);
            if let Ok(value) = value.parse::<f64>() {
                training_data_percent = value / 100.0;
            }
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
        DatasetType::Digits => CompleteDatasetConfig {
            title: "1️⃣  Digits".to_owned(),
            width: 28,
            height: 28,
            split_chars: DataParser::NEW_LINE_CHARS.to_vec(),
            char_range: vec![' ', '+', '#'],
            training_data_path: "data/digitdata/trainingimages".to_owned(),
            training_labels_path: "data/digitdata/traininglabels".to_owned(),
            validation_data_path: "data/digitdata/testimages".to_owned(),
            validation_labels_path: "data/digitdata/testlabels".to_owned(),
            label_range: 10,
            label_to_text_fn: Box::new(&|label| label.to_string()),
        },
        DatasetType::Faces => CompleteDatasetConfig {
            title: "😀 Faces".to_owned(),
            width: 60,
            height: 70,
            split_chars: DataParser::NEW_LINE_CHARS.to_vec(),
            char_range: vec![' ', '#'],
            training_data_path: "data/facedata/facedatatrain".to_owned(),
            training_labels_path: "data/facedata/facedatatrainlabels".to_owned(),
            validation_data_path: "data/facedata/facedatatest".to_owned(),
            validation_labels_path: "data/facedata/facedatatestlabels".to_owned(),
            label_range: 2,
            label_to_text_fn: Box::new(&|label| {
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
                "   RNG seed: {}\n   Layers: {:?}\n   Epochs: {}\n   Learn rate: {}",
                seed, layers, epochs, learn_rate
            );
            ai_name = "Neural network".to_owned();
            ai = Box::new(NeuralNetwork::from_structure(
                seed,
                &layers,
                activation_func,
                activation_func_deriv,
                move |_| learn_rate,
            ))
        }
        AiType::Perceptron => {
            println!("👀 Perceptron");
            println!("   Epochs: {}\n   Learn rate: {}", epochs, learn_rate);

            ai_name = "Perceptron".to_owned();
            ai = Box::new(Perceptron::new(learn_rate))
        }
    };

    let ai_config = AIConfig {
        ai,
        epochs,
        name: ai_name,
    };

    let mut runner: Box<dyn Runner> = match runner_type {
        RunnerType::Manual => Box::new(ManualRunner {
            ai_config,
            dataset_config,
            training_data_percent,
        }),
        RunnerType::Sampler => Box::new(SamplerRunner {
            ai_config,
            dataset_config,
            seed,
            sample_count,
            training_data_percent,
        }),
        RunnerType::SamplerSet => Box::new(SamplerSetRunner {
            ai_config,
            dataset_config,
            sample_count,
            seed,
            training_data_percents: vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }),
    };

    runner.run();
}
