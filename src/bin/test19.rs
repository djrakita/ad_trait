use num_traits::real::Real;
use ad_trait::AD;
use ad_trait::differentiable_block::{DifferentiableBlock, DifferentiableBlockTrait, ForwardADMulti, Spider2ForwardAD};

use ad_trait::forward_ad::adfn::adfn;

pub struct Test;
impl DifferentiableBlockTrait for Test {
    type U<T: AD> = ();

    fn call<T1: AD>(inputs: &[T1], _args: &Self::U<T1>) -> Vec<T1> {
        vec![ inputs[0] + inputs[1].sin() + inputs[2] ]
    }

    fn num_inputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        3
    }

    fn num_outputs<T1: AD>(_args: &Self::U<T1>) -> usize {
        1
    }
}

fn main() {
    let derivative_data = Spider2ForwardAD::<Test, adfn<1>>::new(&(), 0.1, true);
    // let derivative_data = Ricochet::<Test, adfn<1>>::new(&(), RicochetTermination::MaxIters(1));
    let d1 = DifferentiableBlock::new(derivative_data);

    let derivative_data = ForwardADMulti::<Test, adfn<4>>::new();
    let d2 = DifferentiableBlock::new(derivative_data);

    let mut inputs = vec![1.0, 2.0, 3.0];
    for i in 0..5 {
        let a = i as f64 * 0.001;
        inputs[0] += a;
        inputs[1] += a*0.1;
        inputs[2] -= a;
        let res1 = d1.derivative(&inputs, &());
        let res2 = d2.derivative(&inputs, &());

        println!("{:?}", res1);
        println!("{:?}", res2);

        let dot = res1.1.dot(&res2.1);
        assert!(dot >= 0.0, "dot must be positive: {}", dot);
        println!("{}", dot);
        println!("------");
    }
}