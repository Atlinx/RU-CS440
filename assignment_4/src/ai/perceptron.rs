use std::{borrow::Borrow, string, sync::Arc};

use crate::{
    ai::{label_certainty_from_vec, perceptron},
    types::{FeaturesVec, LabelFeatureVecDataPoint, LabelVec},
};

use super::{activation_func, AI};

#[derive(Clone)]
pub struct Perceptron {
    learn_rate: f64,
    weights_of_all_perceptrons: Vec<Vec<f64>>,
    label_range: usize,
    feature_size: usize,
}

impl Perceptron {
    pub fn new(learn_rate: f64, label_range: usize, feature_size: usize) -> Self {
        let mut inst = Perceptron {
            learn_rate,
            weights_of_all_perceptrons: Vec::new(),
            label_range,
            feature_size,
        };
        inst.reset();
        inst
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
        let (label_index, _) = label_certainty_from_vec(label);
        let mut guesser = 0;
        let mut guessers_guess = 0.0;

        for perceptron_index in 0..self.weights_of_all_perceptrons.len() {
            let weights = &self.weights_of_all_perceptrons[perceptron_index];
            let mut f = 0.0;
            for i in 0..features.len() {
                f += weights[i] * features.get((i, 0)).unwrap();
            }
            f += weights[weights.len() - 1];
            if f >= guessers_guess {
                guessers_guess = f;
                guesser = perceptron_index;
            }
        }
        if guesser == label_index {
            return true;
        }

        let guessers_weights = &mut self.weights_of_all_perceptrons[guesser];
        let guessers_weights_last = guessers_weights.len() - 1;
        for i in 0..guessers_weights_last {
            guessers_weights[i] -= self.learn_rate * features.get((i, 0)).unwrap();
        }
        guessers_weights[guessers_weights_last] -= self.learn_rate;

        let correct_weights = &mut self.weights_of_all_perceptrons[label_index];
        let correct_weights_last = correct_weights.len() - 1;
        for i in 0..correct_weights_last {
            correct_weights[i] += self.learn_rate * features.get((i, 0)).unwrap();
        }
        correct_weights[correct_weights_last] += self.learn_rate;

        false
    }

    /// Tests the neural network on a data point.
    /// Returns the a tuple containing the prediction label and certainty of the prediction.
    fn test_data_point(&self, features: &FeaturesVec) -> (usize, f64) {
        let mut guesser = 0;
        let mut guessers_guess = 0.0;

        for perceptron_index in 0..self.weights_of_all_perceptrons.len() {
            let weights = &self.weights_of_all_perceptrons[perceptron_index];
            let mut f = 0.0;
            for i in 0..features.len() {
                f += weights[i] * features.get((i, 0)).unwrap();
            }
            f += weights[weights.len() - 1];
            f = activation_func(f);
            if f >= guessers_guess {
                guessers_guess = f;
                guesser = perceptron_index;
            }
        }

        (guesser, guessers_guess)
    }

    /// Resets training
    fn reset(&mut self) {
        let mut weights_set = Vec::new();
        for _ in 0..self.label_range {
            let mut weights: Vec<f64> = Vec::new();
            for _ in 0..(self.feature_size + 1) {
                weights.push(0.0);
            }
            weights_set.push(weights);
        }
        self.weights_of_all_perceptrons = weights_set;
    }
}
