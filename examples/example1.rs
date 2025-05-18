use ad_trait::AD;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::adr;

#[derive(Clone)]
pub struct Test<T: AD> {
    coeff: T
}
impl<T: AD> DifferentiableFunctionTrait<T> for Test<T> {
    const NAME: &'static str = "Test";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![ self.coeff*inputs[0].sin() + inputs[1].cos() ]
    }

    fn num_inputs(&self) -> usize {
        2
    }

    fn num_outputs(&self) -> usize {
        1
    }
}
impl<T: AD> Test<T> {
    pub fn to_other_ad_type<T2: AD>(&self) -> Test<T2> {
        Test { coeff: self.coeff.to_other_ad_type::<T2>() }
    }
}


fn main() {
    let inputs = vec![1., 2.];

    // Reverse AD //////////////////////////////////////////////////////////////////////////////////
    let function_standard = Test { coeff: 2.0 };
    let function_derivative = function_standard.to_other_ad_type::<adr>();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ReverseAD::new());

    let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
    println!("Reverse AD: ");
    println!("  f_res: {}", f_res[0]);
    println!("  derivative: {}", derivative_res);
    println!("//////////////");
    println!();

    // Forward AD //////////////////////////////////////////////////////////////////////////////////
    let function_standard = Test { coeff: 2.0 };
    let function_derivative = function_standard.to_other_ad_type::<adfn<1>>();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardAD::new());

    let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
    println!("Forward AD: ");
    println!("  f_res: {}", f_res[0]);
    println!("  derivative: {}", derivative_res);
    println!("//////////////");
    println!();

    // Forward AD Multi ////////////////////////////////////////////////////////////////////////////
    let function_standard = Test { coeff: 2.0 };
    let function_derivative = function_standard.to_other_ad_type::<adfn<2>>();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardADMulti::new());

    let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
    println!("Forward AD Multi: ");
    println!("  f_res: {}", f_res[0]);
    println!("  derivative: {}", derivative_res);
    println!("//////////////");
    println!();

    // Finite Differencing /////////////////////////////////////////////////////////////////////////
    let function_standard = Test { coeff: 2.0 };
    let function_derivative = function_standard.clone();
    let differentiable_block = FunctionEngine::new(function_standard, function_derivative, FiniteDifferencing::new());

    let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
    println!("Finite Differencing: ");
    println!("  f_res: {}", f_res[0]);
    println!("  derivative: {}", derivative_res);
    println!("//////////////");
    println!();

}