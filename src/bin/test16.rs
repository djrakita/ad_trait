
use nalgebra::DMatrix;
use num_traits::real::Real;
use simba::simd::SimdValue;
use ad_trait::AD;
use ad_trait::differentiable_block::{DifferentiableBlock, DifferentiableBlockTrait, FiniteDifferencing, FiniteDifferencingMulti, ForwardAD, ForwardADMulti, ReverseAD, Ricochet, RicochetData, RicochetTermination};
use ad_trait::forward_ad::adf::{adf_f32x16, adf_f32x2, adf_f32x4, adf_f32x8, adf_f64x2, adf_f64x4, adf_f64x8};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::simd::f64xn::f64xn;

struct Test;
impl DifferentiableBlockTrait for Test {
    type U<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], _args: &Self::U<T1>) -> Vec<T1> {
        vec![inputs[0] * inputs[1] * inputs[2] * inputs[3] * inputs[4] * inputs[5] * inputs[6] * inputs[7]]
    }

    fn num_inputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        8
    }

    fn num_outputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        1
    }
}

fn main() {
    let d1 = DifferentiableBlock::<Test, _, _>::new(ForwardAD::new());
    let derivative_data = Ricochet::<Test, adf_f64x2>::new(&(), RicochetTermination::MaxIters(1));
    let mut d2 = DifferentiableBlock::<Test, _, _>::new(derivative_data);
    let derivative_data = Ricochet::<Test, adf_f64x4>::new(&(), RicochetTermination::MaxIters(1));
    let mut d3 = DifferentiableBlock::<Test, _, _>::new(derivative_data);

    let inputs = vec![1.,2.,3.,1.,1.,1.,1.,1.];
    let res = d1.derivative(&inputs, &());
    d2.derivative_data().ricochet_data().update_previous_derivative(&res.1);
    d3.derivative_data().ricochet_data().update_previous_derivative(&res.1);

    let a = 0.015;
    for i in 0..100 {
        let inputs = vec![
            1. + (i as f64) * a ,
            2.+ (i as f64) * a*0.91,
            3.+ (i as f64) * a*0.41,
            1. + (i as f64) * a*0.11,
            1.+ (i as f64) * a*0.14,
            1.0 + (i as f64) * a,
            1. + (i as f64) * a*0.4,
            1.+ (i as f64) * a*0.33
        ];

        println!("{:?}", inputs);

        let res = d1.derivative(&inputs, &());
        println!("{}", res.1);
        let res = d2.derivative(&inputs, &());
        println!("{}", res.1);
        let res = d3.derivative(&inputs, &());
        println!("{}", res.1);

        println!("-----------");

    }
}