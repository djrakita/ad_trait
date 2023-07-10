use std::time::Instant;
use ad_trait::AD;
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::simd::f64xn::f64xn;
use vek::{Vec16};

fn main() {
    let m = 40;
    let n = 40;
    let q = 40;
    let mnq = m*n*q;
    println!("mnq: {:?}", mnq);

    let v1 = 1.0;
    let v2 = 1.01;
    let mut v3 = 0.0;

    let start = Instant::now();
    for _ in 0..(m*n*q) {
        v3 += v1*v2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", v3);

    const N: usize = 16;
    let v1 = f64xn::<N>::splat(1.0);
    let v2 = f64xn::<N>::splat(1.01);
    let mut v3 = f64xn::<N>::splat(0.0);

    let start = Instant::now();
    for _ in 0..(m*n*q) {
        v3 += v1*v2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", v3);

    let v1 = adfn::<N>::constant(1.0);
    let v2 = adfn::<N>::constant(1.01);
    let mut v3 = adfn::<N>::constant(0.0);

    let start = Instant::now();
    for _ in 0..(m*n*q) {
        v3 += v1*v2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", v3);

    let v1 = Vec16::<f64>::one();
    let v2 = Vec16::<f64>::one();
    let mut v3 = Vec16::<f64>::zero();

    let start = Instant::now();
    for _ in 0..(m*n*q) {
        v3 += v1*v2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", v3);
}