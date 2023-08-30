use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD, ReverseAD};

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
    let res = Test::derivative::<ReverseAD>(&[1.,2.], &(), &());
    println!("{:?}", res);

    let res = Test::derivative::<ForwardAD>(&[1.,2.], &(), &());
    println!("{:?}", res);
}