#![feature(generic_associated_types)]

use num_traits::real::Real;
use ad_trait::AD;
use ad_trait::differentiable_block::{DerivativeTrait, DifferentiableBlockTrait, FlowData, FlowFiniteDiff, FlowForwardAD, FlowForwardADMulti};
use ad_trait::forward_ad::adf::{adf_f32x16, adf_f32x2, adf_f32x4};
use ad_trait::forward_ad::adfn::adfn;

pub struct Test;
impl DifferentiableBlockTrait for Test {
    type U<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], _args: &Self::U<T1>) -> Vec<T1> {
        vec![ inputs[0] * inputs[1] * inputs[2] + inputs[3].sin() + inputs[4].sin() * inputs[5] ]
    }

    fn num_inputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        6
    }

    fn num_outputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        1
    }
}

fn main() {
    let f = FlowForwardADMulti::<Test, adfn<9>>::new(&(), 0.6, 5, 0.2);
    let res = f.derivative(&[1.,2.,3.,4.,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());
    let res = f.derivative(&[1.,2.,3.,4.01,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());
    let res = f.derivative(&[1.,2.,3.,4.11,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());
    let res = f.derivative(&[1.1,2.1,3.,4.11,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());
    let res = f.derivative(&[1.1,2.1,3.4,4.11,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());
    let res = f.derivative(&[1.1,2.3,3.7,4.11,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());
    let res = f.derivative(&[1.11,2.3,3.7,4.11,5.,6.], &());
    println!("{:?}", res);
    println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
    println!("{:?}", f.num_function_calls_on_previous_derivative());

}