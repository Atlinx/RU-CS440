use crate::{
    ai::AI,
    feature_extractor::pixel_feature_extractor::PixelFeatureExtractor,
    parser::{
        data_parser::DataParser, label_data_feature_parser::LabelDataFeatureParser,
        label_parser::LabelParser,
    },
    types::LabelFeatureVecDataSet,
};

pub mod manual_runner;
pub mod sampler_runner;
pub mod sampler_set_runner;

#[derive(Clone)]
pub struct AIConfig {
    pub name: String,
    pub ai: Box<dyn AI>,
    pub epochs: usize,
}

#[derive(Clone)]
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
    pub label_range: usize,
    pub label_to_text_fn: Box<&'static (dyn Fn(usize) -> String + Sync + Send)>,
}

impl CompleteDatasetConfig {
    /// Loads the training anbd test data sets
    /// Returns (training_data_set, )
    fn load_training_test_datasets(&self) -> (LabelFeatureVecDataSet, LabelFeatureVecDataSet) {
        let label_data_parser = LabelDataFeatureParser::new(
            DataParser::new(
                self.width,
                self.height,
                self.split_chars.clone(),
                self.char_range.clone(),
            ),
            PixelFeatureExtractor,
            LabelParser::new(),
        );
        let labelled_training_data = label_data_parser
            .parse_files_to_vec(
                &self.training_data_path,
                &self.training_labels_path,
                self.label_range,
            )
            .expect("Expect label and data files to be parsable");
        let labelled_test_data = label_data_parser
            .parse_files_to_vec(
                &self.validation_data_path,
                &self.validation_labels_path,
                self.label_range,
            )
            .expect("Expect label and data files to be parsable");

        println!("{}", self.title);
        println!("   Training set size: {}", labelled_training_data.len());
        println!("   Test set size: {}", labelled_test_data.len());

        (labelled_training_data, labelled_test_data)
    }
}

pub trait Runner {
    fn run(&mut self);
}
