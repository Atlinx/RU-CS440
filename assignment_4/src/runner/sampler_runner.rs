use std::time::Instant;

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::types::F64VecExtens;

use super::{AIConfig, CompleteDatasetConfig, Runner};

pub struct SamplerRunner {
    pub ai_config: AIConfig,
    pub dataset_config: CompleteDatasetConfig,
    pub seed: u64,
    pub sample_count: usize,
    pub training_data_percent: f64,
}

impl Runner for SamplerRunner {
    fn run(&mut self) {
        println!("🤖 Sampler Runner");
        println!(
            "   Sample count: {}\n   Used training data: {:.2}%",
            self.sample_count,
            self.training_data_percent * 100.0
        );

        let (labelled_training_data, labelled_test_data) =
            self.dataset_config.load_training_test_datasets();
        let used_training_data_size =
            (labelled_training_data.len() as f64 * self.training_data_percent).round() as usize;

        println!("🚄 Running training tests");

        let mut accuracy_vec = Vec::new();
        let mut time_elapsed_vec = Vec::new();
        for _ in 0..self.sample_count {
            let mut rand = StdRng::seed_from_u64(self.seed);
            let used_training_data = labelled_training_data
                .choose_multiple(&mut rand, used_training_data_size)
                .cloned()
                .collect();

            let now = Instant::now();
            self.ai_config
                .ai
                .train_data_set(&used_training_data, self.ai_config.epochs, false);
            let time_elapsed = now.elapsed().as_secs_f64();
            let accuracy = self.ai_config.ai.test_accuracy(&labelled_test_data);
            time_elapsed_vec.push(time_elapsed);
            accuracy_vec.push(accuracy);
            println!(
                "   Accuracy: {:.2}% Time elapsed: {:.4}s",
                accuracy * 100.0,
                time_elapsed
            );
            self.ai_config.ai.reset();
        }
        let error_vec: Vec<f64> = accuracy_vec.iter().cloned().map(|x| 1.0 - x).collect();
        let (accuracy_mean, accuracy_stdev) = accuracy_vec.dist_info();
        let (error_mean, error_stdev) = error_vec.dist_info();
        let (time_elapsed_mean, time_elapsed_stdev) = time_elapsed_vec.dist_info();
        println!(
            "📄 {} Accuracy: (Mean: {:.2}%, Stdev: {:.2}%) Error: (Mean: {:.2}%, Stdev: {:.2}%) Time: (Mean: {:.4}s, Stdev: {:.4}s)",
            self.ai_config.name,
            accuracy_mean * 100.0,
            accuracy_stdev * 100.0,
            error_mean * 100.0,
            error_stdev * 100.0,
            time_elapsed_mean,
            time_elapsed_stdev,
        );
    }
}
