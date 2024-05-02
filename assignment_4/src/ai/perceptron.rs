use crate::types::{FeaturesVec, LabelFeatureVecDataSetSlice, LabelVec};

use super::{label_certainty_from_vec, AI};

pub struct Perceptron {
    learn_rate: f64,
}

impl Perceptron {
    pub fn new(learn_rate: f64) -> Self {
        Perceptron { learn_rate }
    }
}

impl AI for Perceptron {
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
        // TODO: Forward propagation

        // TODO: Back propagation

        // TODO: Return whether correct or not
        todo!()
    }

    // Tests the neural network on a data point.
    // Returns the a tuple containing the prediction label and certainty of the prediction.
    fn test_data_point(&self, features: &FeaturesVec) -> (usize, f64) {
        // TODO: Forward propagation
        todo!()
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
