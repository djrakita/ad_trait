use std::time::Instant;
use num_traits::real::Real;
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait, ForwardADMulti};
use ad_trait::forward_ad::adf::{adf_f32x2, adf_f32x4};

pub struct Test;
impl<A: AD> DifferentiableFunctionTrait<A> for Test {
    fn call(&self, inputs: &[A], _freeze: bool) -> Vec<A> {
        return vec![inputs[0].cos() + inputs[1].sin()];
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

pub struct TestC;
impl DifferentiableFunctionClass for TestC {
    type FunctionType<T: AD> = Test;
}

fn main() {

    let d = DifferentiableBlock::new_with_tag(TestC, ForwardADMulti::<adf_f32x2>::new(), Test, Test);

    let start = Instant::now();
    for _ in 0..1000 {
        d.derivative(&[1., 2.]);
    }
    println!("{:?}", start.elapsed());



    let d = DifferentiableBlock::new_with_tag(TestC, ForwardADMulti::<adf_f32x4>::new(), Test, Test);
    let start = Instant::now();
    for _ in 0..1000 {
        d.derivative(&[1., 2.]);
    }
    println!("{:?}", start.elapsed());
}