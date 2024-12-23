use ad_trait::AD;
use ad_trait::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionTrait, WASP};

struct Test;
impl<T: AD> DifferentiableFunctionTrait<T> for Test {
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        return vec![ inputs[0].sin(), inputs[1].cos() ]
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        2
    }
}

fn main() {
    let w = WASP::new(2, 2, true, 0.3, 0.3);
    let res = w.derivative(&[1., 2.], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
    let res = w.derivative(&[1., 2.], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
    let res = w.derivative(&[1., 2.05], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
    let res = w.derivative(&[1., 2.05], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
}