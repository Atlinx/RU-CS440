use std::{error::Error, fs::File, io::Read};

use crate::types::Label;

pub struct LabelParser {}

impl LabelParser {
    pub fn new() -> Self {
        LabelParser {}
    }

    pub fn parse_file(&self, file_path: &str) -> Result<Vec<Label>, Box<dyn Error>> {
        let mut file = File::open(file_path)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let mut labels = Vec::new();
        for label in buffer.split_whitespace() {
            labels.push(label.parse::<Label>()?)
        }
        Ok(labels)
    }
}
