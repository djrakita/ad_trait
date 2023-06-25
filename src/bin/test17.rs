#![feature(generic_associated_types)]

use nalgebra::{DMatrix, DVector};
use num_traits::Pow;
use ad_trait::AD;
use ad_trait::differentiable_block::{DerivativeDataTrait, DifferentiableBlockTrait, SpiderForwardAD};
use ad_trait::forward_ad::adf::adf_f32x2;
use ad_trait::forward_ad::adfn::adfn;

pub struct Test;
impl DifferentiableBlockTrait for Test {
    type U<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], _args: &Self::U<T1>) -> Vec<T1> {
        vec![ inputs[0] + inputs[1] + inputs[2] + inputs[3] ]
    }

    fn num_inputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        4
    }

    fn num_outputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        1
    }
}

fn main() {

    // let s = SpiderData::new(5, 2, 1,-1.0, 1.0);

    /*
    let s = SpiderForwardAD::<Test, adfn<2>>::new(&(), 0.9999999999);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    s.spider_data().print_w();
    */

    // let w = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
    // let wpinv = w.clone().pseudo_inverse(0.0).unwrap();
    // println!("{}", w*wpinv);

    let w = DMatrix::<f64>::from_partial_diagonal(2, 2, &[1.2,5.]);
    let wp = w.clone().pseudo_inverse(0.0).unwrap();

    println!("{}", wp * w);

}