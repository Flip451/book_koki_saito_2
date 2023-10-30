pub struct LearningRate(f64);

impl LearningRate {
    pub fn new(lr: f64) -> Self {
        Self(lr)
    }

    pub fn value(&self) -> f64 {
        self.0
    }
}
