use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, WASP2};

pub struct Test;
impl<T: AD> DifferentiableFunctionTrait<T> for Test {
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![ inputs[0].sin() + inputs[1].cos(), inputs[0].sin() * inputs[1].cos() ]
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        2
    }
}

fn main() {
    let w = WASP2::new(10, 1, 0.01, 10);

    /*
    let res = w.derivative(&[0.,0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[0.,0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[0.,0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[-0.01,0.01], &Test);
    println!("{:?}", res);

    let fd = FiniteDifferencing::new();
    let res = fd.derivative(&[-0.01,0.01], &Test);
    println!("{:?}", res);
    */
}