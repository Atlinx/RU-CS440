use super::{AIConfig, CompleteDatasetConfig, Runner};

pub struct SamplerRunner {
    pub ai_config: AIConfig,
    pub dataset_config: CompleteDatasetConfig,
    pub seed: u64,
}

impl Runner for SamplerRunner {
    fn run(&mut self) {}
}
