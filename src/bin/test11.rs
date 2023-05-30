use std::time::Instant;
use nalgebra::DMatrix;
use ad_trait::AD;
use ad_trait::forward_ad::adf::adf;
use ad_trait::reverse_ad::adr::adr;
use ad_trait::simd::f64xn::f64xn;

fn main() {

    let d1 = DMatrix::<f64>::from_vec(20, 20, vec![1.0; 400]);
    let d2 = DMatrix::<f64>::from_vec(20, 20, vec![1.0; 400]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("{:?}", start.elapsed());

    let d1 = DMatrix::from_vec(20, 20, vec![adr::constant(1.0); 400]);
    let d2 = DMatrix::from_vec(20, 20, vec![adr::constant(1.0); 400]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("{:?}", start.elapsed());

}