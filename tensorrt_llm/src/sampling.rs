use crate::{Error, Result};
use tensorrt_llm_sys as sys;

#[derive(Clone, Debug)]
pub struct SamplingConfig {
    beam_width: i32,
    top_k: Option<i32>,
    top_p: Option<f32>,
    top_p_min: Option<f32>,
    top_p_reset_ids: Option<i32>,
    top_p_decay: Option<f32>,
    temperature: Option<f32>,
    seed: Option<u64>,
    min_tokens: Option<i32>,
    beam_search_diversity_rate: Option<f32>,
    repetition_penalty: Option<f32>,
    presence_penalty: Option<f32>,
    frequency_penalty: Option<f32>,
    prompt_ignore_length: Option<i32>,
    length_penalty: Option<f32>,
    early_stopping: Option<i32>,
    no_repeat_ngram_size: Option<i32>,
    num_return_sequences: Option<i32>,
    min_p: Option<f32>,
    beam_width_array: Option<Vec<i32>>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            beam_width: 1,
            top_k: None,
            top_p: None,
            top_p_min: None,
            top_p_reset_ids: None,
            top_p_decay: None,
            temperature: None,
            seed: None,
            min_tokens: None,
            beam_search_diversity_rate: None,
            repetition_penalty: None,
            presence_penalty: None,
            frequency_penalty: None,
            prompt_ignore_length: None,
            length_penalty: None,
            early_stopping: None,
            no_repeat_ngram_size: None,
            num_return_sequences: None,
            min_p: None,
            beam_width_array: None,
        }
    }
}

impl SamplingConfig {
    pub fn beam_width(mut self, beam_width: i32) -> Self {
        self.beam_width = beam_width;
        self
    }

    pub fn top_k(mut self, top_k: i32) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    pub fn top_p_min(mut self, top_p_min: f32) -> Self {
        self.top_p_min = Some(top_p_min);
        self
    }

    pub fn top_p_reset_ids(mut self, top_p_reset_ids: i32) -> Self {
        self.top_p_reset_ids = Some(top_p_reset_ids);
        self
    }

    pub fn top_p_decay(mut self, top_p_decay: f32) -> Self {
        self.top_p_decay = Some(top_p_decay);
        self
    }

    pub fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn min_tokens(mut self, min_tokens: i32) -> Self {
        self.min_tokens = Some(min_tokens);
        self
    }

    pub fn beam_search_diversity_rate(mut self, beam_search_diversity_rate: f32) -> Self {
        self.beam_search_diversity_rate = Some(beam_search_diversity_rate);
        self
    }

    pub fn repetition_penalty(mut self, repetition_penalty: f32) -> Self {
        self.repetition_penalty = Some(repetition_penalty);
        self
    }

    pub fn presence_penalty(mut self, presence_penalty: f32) -> Self {
        self.presence_penalty = Some(presence_penalty);
        self
    }

    pub fn frequency_penalty(mut self, frequency_penalty: f32) -> Self {
        self.frequency_penalty = Some(frequency_penalty);
        self
    }

    pub fn prompt_ignore_length(mut self, prompt_ignore_length: i32) -> Self {
        self.prompt_ignore_length = Some(prompt_ignore_length);
        self
    }

    pub fn length_penalty(mut self, length_penalty: f32) -> Self {
        self.length_penalty = Some(length_penalty);
        self
    }

    pub fn early_stopping(mut self, early_stopping: i32) -> Self {
        self.early_stopping = Some(early_stopping);
        self
    }

    pub fn no_repeat_ngram_size(mut self, no_repeat_ngram_size: i32) -> Self {
        self.no_repeat_ngram_size = Some(no_repeat_ngram_size);
        self
    }

    pub fn num_return_sequences(mut self, num_return_sequences: i32) -> Self {
        self.num_return_sequences = Some(num_return_sequences);
        self
    }

    pub fn min_p(mut self, min_p: f32) -> Self {
        self.min_p = Some(min_p);
        self
    }

    pub fn beam_width_array<I>(mut self, beam_width_array: I) -> Self
    where
        I: IntoIterator<Item = i32>,
    {
        self.beam_width_array = Some(beam_width_array.into_iter().collect());
        self
    }

    pub(crate) fn validate(&self) -> Result<()> {
        if self.beam_width <= 0 {
            return Err(Error::InvalidArgument("beam_width must be positive".into()));
        }
        for (name, value) in [
            ("top_p", self.top_p),
            ("top_p_min", self.top_p_min),
            ("top_p_decay", self.top_p_decay),
            ("temperature", self.temperature),
            (
                "beam_search_diversity_rate",
                self.beam_search_diversity_rate,
            ),
            ("repetition_penalty", self.repetition_penalty),
            ("presence_penalty", self.presence_penalty),
            ("frequency_penalty", self.frequency_penalty),
            ("length_penalty", self.length_penalty),
            ("min_p", self.min_p),
        ] {
            if let Some(value) = value
                && !value.is_finite()
            {
                return Err(Error::InvalidArgument(format!("{name} must be finite")));
            }
        }
        if let Some(num_return_sequences) = self.num_return_sequences
            && num_return_sequences <= 0
        {
            return Err(Error::InvalidArgument(
                "num_return_sequences must be positive".into(),
            ));
        }
        if let Some(beam_width_array) = self.beam_width_array.as_ref()
            && beam_width_array.iter().any(|beam_width| *beam_width <= 0)
        {
            return Err(Error::InvalidArgument(
                "beam_width_array values must be positive".into(),
            ));
        }
        Ok(())
    }

    pub(crate) fn as_sys(&self) -> sys::SamplingConfig {
        let beam_width_array = self.beam_width_array.as_deref().unwrap_or(&[]);
        sys::SamplingConfig {
            beam_width: self.beam_width,
            has_top_k: i32::from(self.top_k.is_some()),
            top_k: self.top_k.unwrap_or_default(),
            has_top_p: i32::from(self.top_p.is_some()),
            top_p: self.top_p.unwrap_or_default(),
            has_top_p_min: i32::from(self.top_p_min.is_some()),
            top_p_min: self.top_p_min.unwrap_or_default(),
            has_top_p_reset_ids: i32::from(self.top_p_reset_ids.is_some()),
            top_p_reset_ids: self.top_p_reset_ids.unwrap_or_default(),
            has_top_p_decay: i32::from(self.top_p_decay.is_some()),
            top_p_decay: self.top_p_decay.unwrap_or_default(),
            has_seed: i32::from(self.seed.is_some()),
            seed: self.seed.unwrap_or_default(),
            has_temperature: i32::from(self.temperature.is_some()),
            temperature: self.temperature.unwrap_or_default(),
            has_min_tokens: i32::from(self.min_tokens.is_some()),
            min_tokens: self.min_tokens.unwrap_or_default(),
            has_beam_search_diversity_rate: i32::from(self.beam_search_diversity_rate.is_some()),
            beam_search_diversity_rate: self.beam_search_diversity_rate.unwrap_or_default(),
            has_repetition_penalty: i32::from(self.repetition_penalty.is_some()),
            repetition_penalty: self.repetition_penalty.unwrap_or_default(),
            has_presence_penalty: i32::from(self.presence_penalty.is_some()),
            presence_penalty: self.presence_penalty.unwrap_or_default(),
            has_frequency_penalty: i32::from(self.frequency_penalty.is_some()),
            frequency_penalty: self.frequency_penalty.unwrap_or_default(),
            has_prompt_ignore_length: i32::from(self.prompt_ignore_length.is_some()),
            prompt_ignore_length: self.prompt_ignore_length.unwrap_or_default(),
            has_length_penalty: i32::from(self.length_penalty.is_some()),
            length_penalty: self.length_penalty.unwrap_or_default(),
            has_early_stopping: i32::from(self.early_stopping.is_some()),
            early_stopping: self.early_stopping.unwrap_or_default(),
            has_no_repeat_ngram_size: i32::from(self.no_repeat_ngram_size.is_some()),
            no_repeat_ngram_size: self.no_repeat_ngram_size.unwrap_or_default(),
            has_num_return_sequences: i32::from(self.num_return_sequences.is_some()),
            num_return_sequences: self.num_return_sequences.unwrap_or_default(),
            has_min_p: i32::from(self.min_p.is_some()),
            min_p: self.min_p.unwrap_or_default(),
            beam_width_array: beam_width_array.as_ptr(),
            beam_width_array_len: beam_width_array.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_sampling_option_is_encoded_when_set() {
        let sampling = SamplingConfig::default()
            .beam_width(3)
            .top_k(4)
            .top_p(0.9)
            .top_p_min(0.1)
            .top_p_reset_ids(2)
            .top_p_decay(0.5)
            .temperature(0.7)
            .seed(123)
            .min_tokens(2)
            .beam_search_diversity_rate(0.3)
            .repetition_penalty(1.1)
            .presence_penalty(0.2)
            .frequency_penalty(0.4)
            .prompt_ignore_length(5)
            .length_penalty(0.8)
            .early_stopping(1)
            .no_repeat_ngram_size(6)
            .num_return_sequences(2)
            .min_p(0.05)
            .beam_width_array([1, 2, 3]);

        sampling.validate().unwrap();
        let sys = sampling.as_sys();

        assert_eq!(sys.beam_width, 3);
        assert_eq!(sys.has_top_k, 1);
        assert_eq!(sys.has_top_p, 1);
        assert_eq!(sys.has_top_p_min, 1);
        assert_eq!(sys.has_top_p_reset_ids, 1);
        assert_eq!(sys.has_top_p_decay, 1);
        assert_eq!(sys.has_temperature, 1);
        assert_eq!(sys.has_seed, 1);
        assert_eq!(sys.has_min_tokens, 1);
        assert_eq!(sys.has_beam_search_diversity_rate, 1);
        assert_eq!(sys.has_repetition_penalty, 1);
        assert_eq!(sys.has_presence_penalty, 1);
        assert_eq!(sys.has_frequency_penalty, 1);
        assert_eq!(sys.has_prompt_ignore_length, 1);
        assert_eq!(sys.has_length_penalty, 1);
        assert_eq!(sys.has_early_stopping, 1);
        assert_eq!(sys.has_no_repeat_ngram_size, 1);
        assert_eq!(sys.has_num_return_sequences, 1);
        assert_eq!(sys.has_min_p, 1);
        assert_eq!(sys.beam_width_array_len, 3);
    }

    #[test]
    fn finite_validation_covers_float_options() {
        let invalid = [
            SamplingConfig::default().top_p(f32::NAN),
            SamplingConfig::default().top_p_min(f32::NAN),
            SamplingConfig::default().top_p_decay(f32::NAN),
            SamplingConfig::default().temperature(f32::NAN),
            SamplingConfig::default().beam_search_diversity_rate(f32::NAN),
            SamplingConfig::default().repetition_penalty(f32::NAN),
            SamplingConfig::default().presence_penalty(f32::NAN),
            SamplingConfig::default().frequency_penalty(f32::NAN),
            SamplingConfig::default().length_penalty(f32::NAN),
            SamplingConfig::default().min_p(f32::NAN),
        ];

        for sampling in invalid {
            assert!(sampling.validate().is_err());
        }
    }
}
