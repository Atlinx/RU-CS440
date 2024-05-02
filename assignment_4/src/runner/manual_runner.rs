use std::io::stdin;

use crate::{
    ai::label_certainty_from_vec,
    feature_extractor::pixel_feature_extractor::PixelFeatureExtractor,
    parser::{
        data_parser::DataParser, label_data_feature_parser::LabelDataFeatureParser,
        label_parser::LabelParser,
    },
    types::RawDataExtens,
};

use super::{AIConfig, CompleteDatasetConfig, Runner};

pub struct ManualRunner {
    pub ai_config: AIConfig,
    pub dataset_config: CompleteDatasetConfig,
}

impl Runner for ManualRunner {
    fn run(&mut self) {
        let label_data_parser = LabelDataFeatureParser::new(
            DataParser::new(
                self.dataset_config.width,
                self.dataset_config.height,
                self.dataset_config.split_chars.clone(),
                self.dataset_config.char_range.clone(),
            ),
            PixelFeatureExtractor,
            LabelParser::new(),
        );
        let labelled_training_data = label_data_parser
            .parse_files_to_vec(
                &self.dataset_config.training_data_path,
                &self.dataset_config.training_labels_path,
                self.dataset_config.label_range,
            )
            .expect("Expect label and data files to be parsable");
        let labelled_test_data = label_data_parser
            .parse_files_to_vec(
                &self.dataset_config.validation_data_path,
                &self.dataset_config.validation_labels_path,
                self.dataset_config.label_range,
            )
            .expect("Expect label and data files to be parsable");

        println!("{}", self.dataset_config.title);
        println!("   Training set size: {}", labelled_training_data.len());
        println!("   Test set size: {}", labelled_test_data.len());
        println!(
            "   Used training data: {:.2}%",
            self.dataset_config.training_data_percent * 100.0
        );
        let used_training_data_size = (labelled_training_data.len() as f64
            * self.dataset_config.training_data_percent)
            .round() as usize;
        let used_training_data = &labelled_training_data[..used_training_data_size];
        self.ai_config
            .ai
            .train_data_set(used_training_data, self.ai_config.epochs, true);
        let accuracy = self.ai_config.ai.test_accuracy(&labelled_test_data);
        println!(
            "📄 {} accuracy: {:.2}%",
            self.ai_config.name,
            accuracy * 100.0
        );
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
                    (self.dataset_config.label_to_text_fn)(label_index),
                    label_index,
                );
                data.print();

                let (guess_label_index, guess_certainty) =
                    self.ai_config.ai.test_data_point(features);

                println!(
                    "🧠 {} predicts: ({}: {}) with {:.2}% certainty\n",
                    self.ai_config.name,
                    (self.dataset_config.label_to_text_fn)(guess_label_index),
                    guess_label_index,
                    guess_certainty * 100.0
                )
            } else if input.trim() == "q" {
                break;
            }
        }
    }
}
