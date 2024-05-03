use std::{error::Error, sync::Arc};

use ndarray::Array2;

use crate::{
    feature_extractor::FeatureExtractor,
    types::{
        Features, FeaturesVec, Label, LabelFeatureDataSet, LabelFeatureVecDataSet, LabelVec,
        RawData,
    },
};

use super::{data_parser::DataParser, label_parser::LabelParser};

pub struct LabelDataFeatureParser {
    pub data_parser: DataParser,
    pub feature_extractor: Box<dyn FeatureExtractor>,
    pub label_parser: LabelParser,
}

#[derive(Debug)]
pub enum LabelDataFeatureParserError {
    DataParserError(std::io::Error),
    LabelParserError(Box<dyn Error>),
    FeatureExtractorError(Box<dyn Error>),
    MismatchedSize(usize, usize),
}

impl LabelDataFeatureParser {
    pub fn new<T: FeatureExtractor + 'static>(
        data_parser: DataParser,
        feature_extractor: T,
        label_parser: LabelParser,
    ) -> Self {
        LabelDataFeatureParser {
            data_parser,
            feature_extractor: Box::new(feature_extractor),
            label_parser,
        }
    }

    pub fn parse_files(
        &self,
        data_file_path: &str,
        label_file_path: &str,
    ) -> Result<LabelFeatureDataSet, LabelDataFeatureParserError> {
        let raw_data = self
            .data_parser
            .parse_file(data_file_path)
            .map_err(|e| LabelDataFeatureParserError::DataParserError(e))?;
        let labels = self
            .label_parser
            .parse_file(label_file_path)
            .map_err(|e| LabelDataFeatureParserError::LabelParserError(e))?;
        if raw_data.len() != labels.len() {
            return Err(LabelDataFeatureParserError::MismatchedSize(
                raw_data.len(),
                labels.len(),
            ));
        }
        let mut features_list = Vec::new();
        for data in raw_data.iter() {
            let features = self
                .feature_extractor
                .extract_features(data.clone())
                .map_err(|e| LabelDataFeatureParserError::FeatureExtractorError(e))?;
            features_list.push(features);
        }
        Ok(raw_data
            .into_iter()
            .zip(features_list)
            .zip(labels)
            .map(|((x, y), z)| (x, y, z))
            .collect())
    }

    pub fn parse_files_to_vec(
        &self,
        data_file_path: &str,
        label_file_path: &str,
        label_range: usize,
    ) -> Result<LabelFeatureVecDataSet, LabelDataFeatureParserError> {
        let data_set = self.parse_files(data_file_path, label_file_path)?;
        Ok(self.convert_to_vec(data_set, label_range))
    }

    pub fn convert_to_vec(
        &self,
        data: Vec<(RawData, Features, Label)>,
        label_range: usize,
    ) -> LabelFeatureVecDataSet {
        data.into_iter()
            .map(|(data, features, label)| {
                let feat_vec: FeaturesVec = Array2::from_shape_vec((features.len(), 1), features)
                    .expect("Expect conversion to workfrom_shape");
                let feat_label: LabelVec =
                    Array2::from_shape_fn(
                        (label_range, 1),
                        |(i, _)| {
                            if i == label {
                                1.0
                            } else {
                                0.0
                            }
                        },
                    );
                Arc::new((data, feat_vec, feat_label))
            })
            .collect()
    }
}
