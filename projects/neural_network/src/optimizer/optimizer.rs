pub trait Optimizer<P, G> {
    fn update(&self, params: &mut P, grads: &G);
}
