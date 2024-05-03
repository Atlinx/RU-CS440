use std::{thread, time::Instant};

use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};

use crate::types::F64VecExtens;

use super::{AIConfig, CompleteDatasetConfig, Runner};

pub struct SamplerSetRunner {
    pub ai_config: AIConfig,
    pub dataset_config: CompleteDatasetConfig,
    pub seed: u64,
    pub sample_count: usize,
    pub training_data_percents: Vec<f64>,
}

impl Runner for SamplerSetRunner {
    fn run(&mut self) {
        println!("📐 Sampler Set Runner");
        println!(
            "   Sample count: {}\n   Data % set: {:?}",
            self.sample_count, self.training_data_percents
        );

        let (labelled_training_data, labelled_test_data) =
            self.dataset_config.load_training_test_datasets();

        println!("🚄 Running training tests");

        let mut threads = Vec::new();
        for percent in self.training_data_percents.iter() {
            let mut ai_config = self.ai_config.clone();
            let sample_count = self.sample_count;
            let seed = self.seed;
            let labelled_test_data = labelled_test_data.clone();
            let labelled_training_data = labelled_training_data.clone();
            let percent = *percent;
            let thread = thread::spawn(move || {
                let used_training_data_size =
                    (labelled_training_data.len() as f64 * percent).round() as usize;

                let mut accuracy_vec = Vec::new();
                let mut time_elapsed_vec = Vec::new();
                println!("   Start test for {:.2}% of data", percent * 100.0);
                for _ in 0..sample_count {
                    let mut rand = StdRng::seed_from_u64(seed);
                    let used_training_data = labelled_training_data
                        .choose_multiple(&mut rand, used_training_data_size)
                        .cloned()
                        .collect();

                    let now = Instant::now();
                    ai_config
                        .ai
                        .train_data_set(&used_training_data, ai_config.epochs, false);
                    let time_elapsed = now.elapsed().as_secs_f64();
                    let accuracy = ai_config.ai.test_accuracy(&labelled_test_data);
                    time_elapsed_vec.push(time_elapsed);
                    accuracy_vec.push(accuracy);
                    println!(
                        "   {:.2}% of data -> Accuracy: {:.2}% Time elapsed: {:.4}s",
                        percent * 100.0,
                        accuracy * 100.0,
                        time_elapsed
                    );
                    ai_config.ai.reset();
                }
                let error_vec: Vec<f64> = accuracy_vec.iter().cloned().map(|x| 1.0 - x).collect();
                let (accuracy_mean, accuracy_stdev) = accuracy_vec.dist_info();
                let (error_mean, error_stdev) = error_vec.dist_info();
                let (time_elapsed_mean, time_elapsed_stdev) = time_elapsed_vec.dist_info();
                println!(
					"📄 {:.2}% of data > Accuracy: (Mean: {:.2}%, Stdev: {:.2}%) Error: (Mean: {:.2}%, Stdev: {:.2}%) Time: (Mean: {:.4}s, Stdev: {:.4}s)",
					percent * 100.0,
					accuracy_mean * 100.0,
					accuracy_stdev * 100.0,
					error_mean * 100.0,
					error_stdev * 100.0,
					time_elapsed_mean,
					time_elapsed_stdev,
				);
                (
                    (accuracy_mean, accuracy_stdev),
                    (error_mean, error_stdev),
                    (time_elapsed_mean, time_elapsed_stdev),
                )
            });
            threads.push(thread);
        }

        let mut average_accuracies = Vec::new();
        for thread in threads {
            let average_accuracy = thread.join().expect("Expected join to work");
            average_accuracies.push(average_accuracy);
        }

        println!("📝 Summary");
        println!(
            "   AI: {}\n   Data: {}\n   Sample count: {}",
            self.ai_config.name, self.dataset_config.title, self.sample_count
        );
        println!("🎯 Measurements:");
        for (
            percent,
            (
                (accuracy_mean, accuracy_stdev),
                (error_mean, error_stdev),
                (time_elapsed_mean, time_elapsed_stdev),
            ),
        ) in self
            .training_data_percents
            .iter()
            .zip(average_accuracies.iter())
        {
            println!(
                "   {:.2}% of data > Accuracy: (Mean: {:.2}%, Stdev: {:.2}%) Error: (Mean: {:.2}%, Stdev: {:.2}%) Time: (Mean: {:.4}s, Stdev: {:.4}s)",
				percent * 100.0,
				accuracy_mean * 100.0,
				accuracy_stdev * 100.0,
				error_mean * 100.0,
				error_stdev * 100.0,
				time_elapsed_mean,
				time_elapsed_stdev,
            );
        }
    }
}
