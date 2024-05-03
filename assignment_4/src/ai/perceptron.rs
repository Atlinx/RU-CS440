use std::sync::Arc;

use crate::types::{FeaturesVec, LabelFeatureVecDataPoint, LabelVec};

use super::AI;

#[derive(Clone)]
pub struct Perceptron {
    learn_rate: f64,
}

impl Perceptron {
    pub fn new(learn_rate: f64) -> Self {
        Perceptron { learn_rate }
    }
}

impl AI for Perceptron {
    /// Trains the neural network on an entire data set
    fn train_data_set<'a>(
        &mut self,
        dataset: &Vec<Arc<LabelFeatureVecDataPoint>>,
        epochs: usize,
        print: bool,
    ) {
        if print {
            println!(
                "✏️  Start training on dataset (size: {}, epochs: {})",
                dataset.len(),
                epochs
            )
        }
        for epoch in 0..epochs {
            let mut number_correct: usize = 0;
            for datapoint in dataset {
                let (_, features, label) = datapoint.as_ref();
                let correct = self.train_data_point(&features, &label);
                if correct {
                    number_correct += 1;
                }
            }
            if print {
                println!(
                    "   📆 Epoch {}: Accuracy: {:.2}% Learn rate: {}",
                    epoch,
                    (number_correct as f64 / dataset.len() as f64 * 100.0).round(),
                    self.learn_rate
                )
            }
        }
        if print {
            println!("🏁 Finished training")
        }
    }

    /// Trains the neural network on a single data point.
    /// Returns the whether the neural network correctly predicted
    /// the data point or not.
    fn train_data_point(&mut self, features: &FeaturesVec, label: &LabelVec) -> bool {
        // TODO: Forward propagation

        // TODO: Back propagation

        // TODO: Return whether correct or not
        todo!()
    }

    /// Tests the neural network on a data point.
    /// Returns the a tuple containing the prediction label and certainty of the prediction.
    fn test_data_point(&self, features: &FeaturesVec) -> (usize, f64) {
        // TODO: Forward propagation
        todo!()
    }

    /// Resets training
    fn reset(&mut self) {}
}
