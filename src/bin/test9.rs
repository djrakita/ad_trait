use num_traits::real::Real;
use rand::{Rng, thread_rng};
use ad_trait::AD;
use ad_trait::differentiable_block::{DifferentiableBlock, DifferentiableBlockTrait, FiniteDifferencing, FiniteDifferencingMulti, ForwardAD, ForwardADMulti, ReverseAD, RicochetData};
use simba::scalar::ComplexField;

pub struct Test;
impl DifferentiableBlockTrait for Test {
    type U = ();

    fn call<T: AD>(inputs: &[T], _args: &Self::U) -> Vec<T> {
        return vec![ inputs[0] + inputs[1] + inputs[2] ]
    }

    fn num_outputs(_args: &Self::U) -> usize {
        1
    }
}

pub struct Test2;
impl DifferentiableBlockTrait for Test2 {
    type U = ();

    fn call<T: AD>(inputs: &[T], _args: &Self::U) -> Vec<T> {
        return vec![ inputs[0] + inputs[1] + inputs[2].sin(), inputs[0] * inputs[1] * inputs[2].exp() ]
    }

    fn num_outputs(_args: &Self::U) -> usize {
        2
    }
}

fn main() {
    type D = Test2;
    const K: usize = 16;

    let dd1 = FiniteDifferencingMulti::<D, K>::new();
    let dd2 = FiniteDifferencing::<D>::new();
    let dd3 = ReverseAD::<D>::new();
    let dd4 = ForwardAD::<D>::new();
    let dd5 = ForwardADMulti::<D, K>::new();

    let d = DifferentiableBlock::<D, _>::new(dd2);
    let res = d.derivative(&[1.,2.,3.], &());
    println!("{}", res.1);

    RicochetData::new(10, 3, 2, -0.00001, 0.00001, None);

}