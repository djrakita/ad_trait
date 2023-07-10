
use std::time::Instant;
use num_traits::real::Real;
use simba::simd::{f32x16, SimdComplexField, SimdValue};

fn main() {
    let a = 1.01_f64;
    let mut b = 0.0_f64;
    let start = Instant::now();
    for _ in 0..1000 {
        b += a.powf(2.0);
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", b);


    let a = f32x16::splat(1.01);
    let n = f32x16::splat(2.0);
    let mut b = f32x16::splat(0.0);
    let start = Instant::now();
    for _ in 0..1000 {
        b += a.simd_powf(n);
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", b);
}