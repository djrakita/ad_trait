use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adf::adf_f32x8;

pub struct TestClass;
impl DifferentiableFunctionClass for TestClass {
    type FunctionType<'a, T: AD> = Test;
}

pub struct Test;
impl<'a, T: AD> DifferentiableFunctionTrait<'a, T> for Test {
    fn call(&self, inputs: &[T], _frozen: bool) -> Vec<T> {
        vec![inputs[0].powf(inputs[1])]
        // vec![inputs[0].ln()]
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn main() {
    let db1 = DifferentiableBlock::new_with_tag(TestClass, FiniteDifferencing::new(), Test, Test);
    let db2 = DifferentiableBlock::new_with_tag(TestClass, ForwardAD::new(), Test, Test);
    let db3 = DifferentiableBlock::new_with_tag(TestClass, ReverseAD::new(), Test, Test);
    let db4 = DifferentiableBlock::new_with_tag(TestClass, ForwardADMulti::<adf_f32x8>::new(), Test, Test);

    let input = vec![-1.5, 2.0];

    println!("{:?}", db1.derivative(&input));
    println!("{:?}", db2.derivative(&input));
    println!("{:?}", db3.derivative(&input));
    println!("{:?}", db4.derivative(&input));
}