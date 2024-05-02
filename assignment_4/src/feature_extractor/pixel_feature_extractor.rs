use std::error::Error;

use crate::types::{Features, RawData};

use super::FeatureExtractor;

pub struct PixelFeatureExtractor;

impl FeatureExtractor for PixelFeatureExtractor {
    fn extract_features(&self, data: RawData) -> Result<Features, Box<dyn Error>> {
        // Get a (# pixels) x 1 vector
        Ok(data.array)
    }
}
