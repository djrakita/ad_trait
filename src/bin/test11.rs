use std::time::Instant;
use nalgebra::DMatrix;
use ad_trait::AD;
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::adr;
use ad_trait::simd::f64xn::f64xn;
use vek::Vec32;
use simba::simd::{f64x4, SimdValue, f64x8, f64x2, f32x8, f32x4, f32x16};
use ad_trait::forward_ad::adf::{adf_f32x16, adf_f32x2, adf_f32x4, adf_f32x8, adf_f64x2, adf_f64x4};

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