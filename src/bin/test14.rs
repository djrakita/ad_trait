use ad_trait::AD;
use ad_trait::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionTrait, WASP, WASP2};

struct Test;
impl<T: AD> DifferentiableFunctionTrait<T> for Test {
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        return vec![ inputs[0].sin(), inputs[1].cos() + inputs[2].cos() ]
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn num_outputs(&self) -> usize {
        2
    }
}

fn main() {
    let w = WASP2::new(3, 2, 0.98, true, 0.3, 0.3);
    let res = w.derivative(&[1., 2., 2.], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
    let res = w.derivative(&[1., 2., 2.], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
    let res = w.derivative(&[1., 2.05, 2.], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
    let res = w.derivative(&[1., 2.05, 2.], &Test);
    println!("{:?}, {:?}", res, w.num_f_calls());
}