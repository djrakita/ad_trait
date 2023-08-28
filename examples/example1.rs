use ad_trait::AD;
use ad_trait::differentiable_block::{DifferentiableBlock, DifferentiableFunctionTrait, ForwardADMulti, ReverseAD};
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
    let differentiable_block = DifferentiableBlock::<Test, ReverseAD<_>>::new();
    let f = differentiable_block.call(&[1.0, 2.0], &());
    println!("{:?}", f);
    let (f, df_dx) = differentiable_block.derivative(&[1.0, 2.0], &());
    println!("{:?}", (f, df_dx));

    println!("---");

    let differentiable_block = DifferentiableBlock::<Test, ForwardADMulti<_, adf_f32x16>>::new();
    let f = differentiable_block.call(&[1.0, 2.0], &());
    println!("{:?}", f);
    let (f, df_dx) = differentiable_block.derivative(&[1.0, 2.0], &());
    println!("{:?}", (f, df_dx));
}