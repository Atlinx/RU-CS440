use std::{sync::Arc, vec};

use ndarray::prelude::*;
use ndarray_rand::{rand_distr::Uniform, RandomExt};
use rand::{rngs::StdRng, SeedableRng};

use crate::types::{FeaturesVec, LabelFeatureVecDataPoint, LabelVec};

use super::{label_certainty_from_vec, AI};

#[allow(dead_code)]
#[derive(Clone)]
pub struct NeuralNetwork {
    rng: StdRng,
    weights: Vec<Array2<f64>>,
    biases: Vec<Array2<f64>>,
    layers: Vec<usize>,
    activation_func: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
    activation_func_deriv: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
    learn_rate_func: Arc<dyn Fn(f64) -> f64 + Send + Sync>,
    learn_rate: f64,
}

#[allow(dead_code)]
impl NeuralNetwork {
    pub fn from_structure<
        AF: Fn(f64) -> f64 + 'static + Send + Sync,
        AFD: Fn(f64) -> f64 + 'static + Send + Sync,
        LRF: Fn(f64) -> f64 + 'static + Send + Sync,
    >(
        seed: u64,
        layers: &[usize],
        activation_func: AF,
        activation_func_deriv: AFD,
        learn_rate_func: LRF,
    ) -> Self {
        let mut neural_network = NeuralNetwork {
            rng: StdRng::seed_from_u64(seed),
            weights: Vec::new(),
            biases: Vec::new(),
            layers: layers.iter().copied().collect(),
            activation_func: Arc::new(activation_func),
            activation_func_deriv: Arc::new(activation_func_deriv),
            learn_rate: learn_rate_func(0.0),
            learn_rate_func: Arc::new(learn_rate_func),
        };
        neural_network.reset();
        neural_network
    }

    /// Returns number of layers in the network
    fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns number of nodes in the network
    fn nodes(&self) -> usize {
        let mut total = 0;
        for nodes in self.layers.iter() {
            total += nodes;
        }
        total
    }
}

impl AI for NeuralNetwork {
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
            self.learn_rate = (self.learn_rate_func)(epoch as f64);
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
        // Forward propagation
        let mut a = features.clone();
        let mut layer_values = vec![a.clone()];
        for i in 0..self.weights.len() {
            let bias = &self.biases[i];
            let weights = &self.weights[i];
            a = bias + weights.dot(&a);
            a.mapv_inplace(self.activation_func.as_ref());
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
            layer_values[i].mapv_inplace(self.activation_func_deriv.as_ref());
            delta = (self.weights[i].t().dot(&delta)) * &layer_values[i];
        }

        correct
    }

    /// Tests the neural network on a data point.
    /// Returns the a tuple containing the prediction label and certainty of the prediction.
    fn test_data_point(&self, features: &FeaturesVec) -> (usize, f64) {
        let mut a = features.clone();
        for i in 0..self.weights.len() {
            let bias = &self.biases[i];
            let weights = &self.weights[i];
            let a_pre = bias + weights.dot(&a);
            a = a_pre.mapv_into(self.activation_func.as_ref());
        }
        label_certainty_from_vec(&a)
    }

    /// Resets training
    fn reset(&mut self) {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        for i in 0..(self.layers.len() - 1) {
            let weight = Array2::<f64>::random_using(
                (self.layers[i + 1], self.layers[i]),
                Uniform::new(-0.5, 0.5),
                &mut self.rng,
            );
            let bias = Array2::<f64>::zeros((self.layers[i + 1], 1));
            weights.push(weight);
            biases.push(bias);
        }
        self.weights = weights;
        self.biases = biases;
    }
}
