use std::time::Instant;
use num_traits::Zero;
use simba::scalar::ComplexField;
use ad_trait::forward_ad::adf::adf;
use ad_trait::forward_ad::adfg::adfg;

const D: usize = 16;

fn main() {
    let a = adf::new(1.1, [1.0; D]);
    let b = adf::new(2.0, [1.0; D]);
    let mut c = adf::new(2.0, [1.0; D]);

    let start = Instant::now();
    for _ in 0..1000 {
        c /= a / b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c.value());
    println!("{:?}", c.tangent());

    let a = adfg::new(1.1, [1.0; D]);
    let b = adfg::new(2.0, [1.0; D]);
    let mut c = adfg::new(2.0, [1.0; D]);

    let start = Instant::now();
    for _ in 0..1000 {
        c /= a / b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c.value());
    println!("{:?}", c.tangent());

    let a = adf::new(1.1, [1.0; D]);
    let b = adf::new(2.0, [1.0; D]);
    let mut c = adf::new(2.0, [1.0; D]);

    let aa = adfg::new(a, [a; D]);
    let bb = adfg::new(b, [b; D]);
    let mut cc = adfg::new(c, [c; D]);

    let start = Instant::now();
    for _ in 0..1000 {
        cc /= aa / bb;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", cc.value());
    println!("{:?}", cc.tangent());

    let a = 1.1;
    let b = 2.0;
    let mut c = 2.0;

    let start = Instant::now();
    for _ in 0..1000 {
        c /= a / b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);

    let a = adf::new(1.1, [1.0; D]);
}