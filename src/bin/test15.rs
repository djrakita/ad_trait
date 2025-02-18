use num_traits::real::Real;
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait, ForwardAD, ReverseAD};

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
    let fe = DifferentiableBlock::new_with_tag(TestC, ReverseAD::new(), Test, Test);
    let res = fe.derivative(&[0.1, 0.2]);
    println!("{:?}", res);

    let fe = DifferentiableBlock::new_with_tag(TestC, ForwardAD::new(), Test, Test);
    let res = fe.derivative(&[0.1, 0.2]);
    println!("{:?}", res);
}