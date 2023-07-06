#![feature(generic_associated_types)]

use num_traits::real::Real;
use ad_trait::AD;
use ad_trait::differentiable_block::{DerivativeTrait, DifferentiableBlockTrait, FlowData, FlowFiniteDiff, FlowForwardAD, FlowForwardADMulti, ForwardADMulti};
use ad_trait::forward_ad::adf::{adf_f32x16, adf_f32x2, adf_f32x4};
use ad_trait::forward_ad::adfn::adfn;

pub struct Test;
impl DifferentiableBlockTrait for Test {
    type U<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], _args: &Self::U<T1>) -> Vec<T1> {
        vec![ inputs[0].powi(2) + inputs[1] + inputs[2].sin() + inputs[3] + inputs[4] * inputs[5] ]
    }

    fn num_inputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        6
    }

    fn num_outputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        1
    }
}

fn main() {
    let f = FlowForwardADMulti::<Test, adfn<5>>::new(&(), 0.7, 1, 0.1);
    let f2 = ForwardADMulti::<Test, adfn<4>>::new();
    let mut inputs = vec![1.,2.,3.,4.,5.,6.];
    for i in 0..1000 {
        let a = (i as f64) * 0.01;
        inputs[0] += a;
        inputs[3] += a;
        inputs[5] -= a;

        let res = f.derivative(&inputs, &());
        let res2 = f2.derivative(&inputs, &());
        println!("{:?}", res.0);
        println!("{:?}", res.1);
        println!("{:?}", res2.1);
        let dot = res.1.dot(&res2.1);
        assert!(dot > 0.0);
        println!("dot: {:?}", dot);
        println!("{:?}", f.num_function_calls_on_previous_derivative());
        println!("{:?}", f.max_test_error_ratio_dis_from_1_on_previous_derivative());
        println!("---");
    }
}