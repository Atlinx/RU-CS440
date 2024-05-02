use crate::ai::AI;

pub mod manual_runner;
pub mod sampler_runner;

pub struct AIConfig {
    pub name: String,
    pub ai: Box<dyn AI>,
    pub epochs: usize,
}

pub struct CompleteDatasetConfig {
    pub title: String,
    pub width: usize,
    pub height: usize,
    pub split_chars: Vec<char>,
    pub char_range: Vec<char>,
    pub training_data_path: String,
    pub training_labels_path: String,
    pub validation_data_path: String,
    pub validation_labels_path: String,
    pub training_data_percent: f64,
    pub label_range: usize,
    pub label_to_text_fn: Box<dyn Fn(usize) -> String + Sync + Send>,
}

pub trait Runner {
    fn run(&mut self);
}
