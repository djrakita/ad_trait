use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ForwardAD, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::adr;

pub struct TestFunctionArgs<T: AD> {
    pub test_argument: T
}

pub struct TestFunction;
impl DifferentiableFunctionTrait for TestFunction {
    type ArgsType<T: AD> = TestFunctionArgs<T>;

    fn call<T1: AD>(inputs: &[T1], args: &Self::ArgsType<T1>) -> Vec<T1> {
        vec![ (inputs[0] + inputs[1]).sin(), (inputs[0] * inputs[1]).cos() * args.test_argument ]
    }

    fn num_inputs<T1: AD>(_args: &Self::ArgsType<T1>) -> usize {
        2
    }

    fn num_outputs<T1: AD>(_args: &Self::ArgsType<T1>) -> usize {
        2
    }
}


fn main() {
    let argsf64 = TestFunctionArgs { test_argument: 1.0 };
    let argsforward = TestFunctionArgs { test_argument: adfn::<1>::new_constant(1.0) };

    let mut d = DifferentiableBlock::<TestFunction, ForwardAD>::new(argsf64, argsforward, ());

    let res = d.call(&[1.0, 2.0]);
    println!("{:?}", res);

    let res = d.derivative(&[1.0, 2.0]);
    println!("{:?}", res);

    d.update_args(|x, y| {
        x.test_argument = 2.0;
        y.test_argument = adfn::<1>::new_constant(2.0)
    });

    let res = d.derivative(&[1.0, 2.0]);
    println!("{:?}", res);
}