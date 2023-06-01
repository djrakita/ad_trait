
use nalgebra::DMatrix;
use num_traits::real::Real;
use simba::simd::SimdValue;
use ad_trait::AD;
use ad_trait::differentiable_block::{DifferentiableBlock, DifferentiableBlockTrait, FiniteDifferencing, FiniteDifferencingMulti, ForwardAD, ForwardADMulti, ReverseAD, Ricochet, RicochetData, RicochetTermination};
use ad_trait::forward_ad::adf::{adf_f32x16, adf_f32x2, adf_f32x4, adf_f32x8, adf_f64x2, adf_f64x8};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::simd::f64xn::f64xn;

struct Test;
impl DifferentiableBlockTrait for Test {
    type U<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], args: &Self::U<T1>) -> Vec<T1> {
        vec![inputs[0].cos() * inputs[1] * inputs[2], inputs[0] * inputs[1].tan() * inputs[2]]
    }

    fn num_inputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        3
    }

    fn num_outputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        2
    }
}

fn main() {
    let derivative_data = Ricochet::<Test, adf_f32x2>::new(&(), RicochetTermination::MaxIters(2));
    let d = DifferentiableBlock::<Test, _, _>::new(derivative_data);
    let res = d.derivative(&[1.,2.,3.], &());
    println!("{}", res.1);
}