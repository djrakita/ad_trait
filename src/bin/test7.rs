use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock2;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait2, FiniteDifferencing2, ForwardAD2, ForwardADMulti2, ReverseAD2};
use ad_trait::forward_ad::adf::adf_f32x8;

pub struct TestClass;
impl DifferentiableFunctionClass for TestClass {
    type FunctionType<'a, T: AD> = Test;
}

pub struct Test;
impl<'a, T: AD> DifferentiableFunctionTrait2<'a, T> for Test {
    fn call(&self, inputs: &[T]) -> Vec<T> {
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
    let db1 = DifferentiableBlock2::new_with_tag(TestClass, FiniteDifferencing2::new(), Test, Test);
    let db2 = DifferentiableBlock2::new_with_tag(TestClass, ForwardAD2::new(), Test, Test);
    let db3 = DifferentiableBlock2::new_with_tag(TestClass, ReverseAD2::new(), Test, Test);
    let db4 = DifferentiableBlock2::new_with_tag(TestClass, ForwardADMulti2::<adf_f32x8>::new(), Test, Test);

    let input = vec![-1.5, 2.0];

    println!("{:?}", db1.derivative(&input));
    println!("{:?}", db2.derivative(&input));
    println!("{:?}", db3.derivative(&input));
    println!("{:?}", db4.derivative(&input));
}