use ad_trait::AD;
use ad_trait::differentiable_function::{derivative_angular_distance, DerivativeMethodTrait, DifferentiableFunctionTrait, ForwardAD, WASP};

pub struct Test;
impl<T: AD> DifferentiableFunctionTrait<T> for Test {
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![ inputs[0].sin() + inputs[1].cos(), inputs[3]*inputs[2].sin() + inputs[4] ]
    }

    fn num_inputs(&self) -> usize {
        5
    }

    fn num_outputs(&self) -> usize {
        2
    }
}

fn main() {
    // let w = WASPNec::new(5, 2, 0.99, true);
    // let w = WASPEc::new(5, 2, 0.9, true, 2, 0.33);
    let w = WASP::new(5, 2, 0.1, 0.1, true);
    println!("{}", w.cache.delta_f_mat_t.lock().unwrap());

    let res = w.derivative(&[1.,2., 0., 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.,2., 0.04, 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.,2., 0.05, 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.,2.04, 0., 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.,2.11, 0., 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.1,2.1, 0., 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.2,2.1, 0., 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.2,2.7, 0.1, 0., 0.], &Test);
    println!("{:?}", res);
    println!("{:?}", w.get_num_f_calls());

    let res = w.derivative(&[1.0,3.0, 0.1, 0., 0.], &Test);
    println!("{}", res.1.transpose());
    println!("{:?}", w.get_num_f_calls());

    let fd = ForwardAD::new();
    let res1 = fd.derivative(&[1.0,3.0, 0.1, 0., 0.], &Test);
    println!("{}", res1.1.transpose());

    println!("{:?}", derivative_angular_distance(&res.1.transpose(), &res1.1.transpose()));
}