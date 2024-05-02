use std::error::Error;

use crate::types::{Features, RawData};

pub mod pixel_feature_extractor;

pub trait FeatureExtractor {
    fn extract_features(&self, data: RawData) -> Result<Features, Box<dyn Error>>;
}
