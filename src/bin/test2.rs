use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardADMulti};
use ad_trait::forward_ad::adf::adf_f32x16;

pub struct Test;
impl DifferentiableFunctionTrait for Test {
    type ArgsType<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], _args: &Self::ArgsType<T1>) -> Vec<T1> {
        let output = (inputs[0] * inputs[1]).sin();

        vec![ output ]
    }

    fn num_inputs<T1: AD>(_args: &Self::ArgsType<T1>) -> usize {
        2
    }

    fn num_outputs<T1: AD>(_args: &Self::ArgsType<T1>) -> usize {
        1
    }
}

fn main() {
    let mut d = DifferentiableBlock::<Test, ForwardADMulti<adf_f32x16>>::new((), (), ());
    let res = d.call(&[1.,2.]);
    println!("{:?}", res);

    d.update_args(|x, y| {*x = (); *y = ()});

    let res = d.derivative(&[1.,2.]);
    println!("{:?}", res);
}