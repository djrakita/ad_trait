use num_traits::real::Real;
use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adf::{adf_f32x8, adf_f64x2};
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
    let args = TestFunctionArgs { test_argument: adr::constant(2.0) };
    let res = TestFunction::derivative::<ReverseAD>(&[1.,2.], &args, &());

    println!("{}", res.1);

    let args = TestFunctionArgs { test_argument: adf_f64x2::constant(2.0) };
    let res = TestFunction::derivative::<ForwardADMulti<adf_f64x2>>(&[1.,2.], &args, &());

    println!("{}", res.1);

    let args = TestFunctionArgs { test_argument: 2.0 };
    let res = TestFunction::derivative::<FiniteDifferencing>(&[1.,2.], &args, &());

    println!("{}", res.1);
}