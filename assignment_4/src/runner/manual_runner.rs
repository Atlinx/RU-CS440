use std::io::stdin;

use crate::{ai::label_certainty_from_vec, types::RawDataExtens};

use super::{AIConfig, CompleteDatasetConfig, Runner};

pub struct ManualRunner {
    pub ai_config: AIConfig,
    pub dataset_config: CompleteDatasetConfig,
    pub training_data_percent: f64,
}

impl Runner for ManualRunner {
    fn run(&mut self) {
        println!("👋 Manual Runner");
        println!(
            "   Used training data: {:.2}%",
            self.training_data_percent * 100.0
        );
        let (labelled_training_data, labelled_test_data) =
            self.dataset_config.load_training_test_datasets();
        let used_training_data_size =
            (labelled_training_data.len() as f64 * self.training_data_percent).round() as usize;
        let used_training_data = labelled_training_data[..used_training_data_size]
            .iter()
            .cloned()
            .collect();
        self.ai_config
            .ai
            .train_data_set(&used_training_data, self.ai_config.epochs, true);
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
                let (data, features, label) = labelled_test_data[index].as_ref();

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
