use std::sync::Arc;

use dyn_clone::{clone_trait_object, DynClone};
use ndarray::Array2;

use crate::types::{FeaturesVec, LabelFeatureVecDataPoint, LabelVec};

pub mod neural_network;
pub mod perceptron;

pub trait AI: DynClone + Send + Sync {
    fn train_data_set(
        &mut self,
        dataset: &Vec<Arc<LabelFeatureVecDataPoint>>,
        epochs: usize,
        print: bool,
    );

    /// Trains the neural network on a single data point.
    /// Returns the whether the neural network correctly predicted
    /// the data point or not.
    fn train_data_point(&mut self, features: &FeaturesVec, label: &LabelVec) -> bool;

    /// Tests the neural network on a data point.
    /// Returns the a tuple containing the prediction label and certainty of the prediction.
    fn test_data_point(&self, features: &FeaturesVec) -> (usize, f64);

    /// Returns the accuracy of the neural network accross an entire data set.
    fn test_accuracy<'a>(&self, dataset: &Vec<Arc<LabelFeatureVecDataPoint>>) -> f64 {
        let mut number_correct: usize = 0;
        for datapoint in dataset {
            let (_, features, label) = datapoint.as_ref();
            let (predicted_label_index, _) = self.test_data_point(&features);
            let correct = predicted_label_index == label_certainty_from_vec(&label).0;
            if correct {
                number_correct += 1;
            }
        }
        number_correct as f64 / dataset.len() as f64
    }

    /// Resets training
    fn reset(&mut self);
}

clone_trait_object!(AI);

pub fn label_certainty_from_vec(array: &Array2<f64>) -> (usize, f64) {
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

pub fn activation_func(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn activation_func_deriv(x: f64) -> f64 {
    x * (1.0 - x)
}
