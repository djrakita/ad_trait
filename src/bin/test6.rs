
use simba::scalar::ComplexField;
use ad_trait::AD;
use ad_trait::forward_ad::adf::adf;
use ad_trait::reverse_ad::adr::{adr};

fn main() {
    let a = adf::new(1.1, [0.0]);
    let b = adf::new(2.0, [1.0]);
    let _c = adf::new(3.0, [0.0]);

    let res = (a).powf(b);
    println!("{:?}", res);

    let a = adr::new_variable(1.1, true);
    let b = adr::constant(2.0);
    let _c = adr::constant(3.0);

    let res = (a).powf(b);
    println!("{:?}", res);
    println!("{:?}", res.get_backwards_mode_grad().wrt(&b))
}