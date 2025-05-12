use std::sync::{Arc, RwLock};
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, ReverseAD};
use ad_trait::reverse_ad::adr::adr;

pub struct Test<T: AD> {
    coeff: T
}
impl<T: AD> DifferentiableFunctionTrait<T> for Test<T> {
    const NAME: &'static str = "Test";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![ self.coeff * inputs[0].cos() ]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn main() {
    let t = Arc::new(RwLock::new(Test { coeff: 2.0 }));
    let t2 = Arc::new(RwLock::new(Test { coeff: adr::constant(2.0) }));

    let db = DifferentiableBlock::new(t.clone(), t2.clone(), ReverseAD::new());

    let res = db.derivative(&[1.0]);
    println!("{:?}", res);

    t.write().unwrap().coeff = 3.0;
    t2.write().unwrap().coeff = adr::constant(3.0);

    let res = db.derivative(&[1.0]);
    println!("{:?}", res);
}