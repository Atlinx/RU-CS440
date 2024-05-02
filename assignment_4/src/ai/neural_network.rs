use std::vec;

use ndarray::prelude::*;
use ndarray_rand::{rand::Rng, rand_distr::Uniform, RandomExt};

use crate::types::{FeaturesVec, LabelFeatureVecDataSetSlice, LabelVec};

use super::{label_certainty_from_vec, AI};

pub struct NeuralNetwork {
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    layers: Vec<usize>,
    activation_func: Box<dyn Fn(f64) -> f64 + Sync + Send>,
    activation_func_deriv: Box<dyn Fn(f64) -> f64 + Sync + Send>,
    learn_rate_func: Box<dyn Fn(f64) -> f64 + Sync + Send>,
    learn_rate: f64,
}

impl NeuralNetwork {
    pub fn from_structure<
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
}

impl AI for NeuralNetwork {
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
}
