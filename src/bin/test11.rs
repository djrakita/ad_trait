use std::time::Instant;
use nalgebra::DMatrix;
use ad_trait::AD;
use ad_trait::forward_ad::adfn::adfn;

use ad_trait::simd::f64xn::f64xn;
use vek::Vec32;
use simba::simd::{SimdValue, f32x16};


fn main() {
    let m = 10;
    let n = 10;

    let d1 = DMatrix::<f64>::from_vec(m, n, vec![1.0; m*n]);
    let d2 = DMatrix::<f64>::from_vec(m, n, vec![1.0; m*n]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("{:?}", start.elapsed());

    let d1 = DMatrix::from_vec(m, n, vec![f64xn::<16>::splat(1.0); m*n]);
    let d2 = DMatrix::from_vec(m, n, vec![f64xn::<16>::splat(1.0); m*n]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("? {:?}", start.elapsed());

    let d1 = DMatrix::from_vec(m, n, vec![Vec32::<f32>::one(); m*n]);
    let d2 = DMatrix::from_vec(m, n, vec![Vec32::<f32>::one(); m*n]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("{:?}", start.elapsed());

    let d1 = DMatrix::from_vec(m, n, vec![f32x16::splat(1.0); m*n]);
    let d2 = DMatrix::from_vec(m, n, vec![f32x16::splat(1.0); m*n]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("? {:?}", start.elapsed());


    let d1 = DMatrix::from_vec(m, n, vec![adfn::<1>::constant(1.0); m*n]);
    let d2 = DMatrix::from_vec(m, n, vec![adfn::<1>::constant(1.0); m*n]);

    let start = Instant::now();
    for _ in 0..1000 {
        &d1*&d2;
    }
    println!("? {:?}", start.elapsed());
}