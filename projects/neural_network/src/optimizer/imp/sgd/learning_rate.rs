pub struct LearningRate(f32);

impl LearningRate {
    pub fn new(lr: f32) -> Self {
        Self(lr)
    }

    pub fn value(&self) -> f32 {
        self.0
    }
}
