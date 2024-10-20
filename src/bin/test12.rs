use nalgebra::DVector;
use ad_trait::AD;
use ad_trait::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionTrait, FiniteDifferencing, wasp_projection, WASPCache, WASPEc, WASPNec};

pub struct Test;
impl<T: AD> DifferentiableFunctionTrait<T> for Test {
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![ inputs[0].sin() + inputs[1].cos() ]
    }

    fn num_inputs(&self) -> usize {
        3
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn main() {
    // let w = WASPNec::new(3, 1, 0.99, true);
    let w = WASPEc::new(3, 1, 0.99, true, 1, 0.3);
    let res = w.derivative(&[1.,2., 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.,2., 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.,2., 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.,2., 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.,2.1, 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.1,2.1, 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.2,2.1, 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.2,2.7, 0.], &Test);
    println!("{:?}", res);

    let res = w.derivative(&[1.0,3.0, 0.], &Test);
    println!("{:?}", res);

    let fd = FiniteDifferencing::new();
    println!("{:?}", fd.derivative(&[1.0,3.0, 0.], &Test));
}