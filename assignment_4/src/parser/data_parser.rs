use std::{fs::File, io::Read};

use crate::types::RawData;

pub struct DataParser {
    width: usize,
    height: usize,
    split_chars: Vec<char>,
    value_char_range: Vec<char>,
}

impl DataParser {
    pub const NEW_LINE_CHARS: [char; 2] = ['\n', '\r'];

    pub fn new(
        width: usize,
        height: usize,
        split_chars: Vec<char>,
        value_char_range: Vec<char>,
    ) -> Self {
        DataParser {
            width,
            height,
            split_chars,
            value_char_range,
        }
    }

    pub fn parse_file(&self, file_path: &str) -> Result<Vec<RawData>, std::io::Error> {
        let mut file = File::open(file_path)?;
        let mut buffer = String::new();
        file.read_to_string(&mut buffer)?;
        let mut samples_vec = Vec::new();
        let mut curr_sample = RawData::new(self.width, self.height);
        let mut x_pos: usize = 0;
        let mut y_pos: usize = 0;
        let mut prev_char_was_split: bool = false;
        for char in buffer.chars() {
            if self.split_chars.contains(&char) {
                prev_char_was_split = true;
                continue;
            } else if prev_char_was_split {
                // We are at a new line
                prev_char_was_split = false;
                y_pos += 1;
                x_pos = 0;
                if y_pos >= self.height {
                    // We've finished parsing one sample
                    // println!("finished parsing one sample:");
                    // print_data(&curr_sample);
                    samples_vec.push(curr_sample);
                    y_pos = 0;
                    curr_sample = RawData::new(self.width, self.height);
                }
            }
            if let Some(index) = self.value_char_range.iter().position(|&x| x == char) {
                let percent = index as f64 / (self.value_char_range.len() - 1) as f64;
                curr_sample.set(x_pos, y_pos, percent);
            }
            x_pos += 1;
        }
        samples_vec.push(curr_sample);
        Ok(samples_vec)
    }
}
