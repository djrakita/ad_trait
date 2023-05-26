use simba::scalar::ComplexField;
use ad_trait::AD;
use ad_trait::forward_ad::adf::adf;
use ad_trait::reverse_ad::adr::{adr, GlobalComputationGraph};

fn main() {
    let a = adf::new(1.0, [1.0]);
    let b = adf::new(2.0, [0.0]);
    let c = adf::new(3.0, [0.0]);

    let res = (a/b).powf(c).sin().cos().sinh().cosh().atanh();
    println!("{:?}", res);


    let a = adr::new(1.0, true);
    let b = adr::constant(2.0);
    let c = adr::constant(3.0);

    let res = (a/b).powf(c).sin().cos().sinh().cosh().atanh();
    println!("{:?}", res);
    println!("{:?}", res.get_backwards_mode_grad().wrt(&a))

}