
use std::time::Instant;
use simba::scalar::ComplexField;
use simba::simd::{f32x16, SimdValue};
use ad_trait::AD;
use ad_trait::forward_ad::adf::{adf_f32x16};
use ad_trait::forward_ad::adfn::adfn;

fn main() {
    let a = 1.0;
    let b = 2.0;
    let mut c = 0.0;
    let start = Instant::now();
    for _ in 0..1000 {
        c += a.sin() * b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);

    let a = adf_f32x16::new(1.0, f32x16::splat(2.0));
    let b  = adf_f32x16::constant(2.0);
    let mut c = adf_f32x16::constant(0.0);
    let start = Instant::now();
    for _ in 0..1000 {
         c += a.sin() * b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);

    let a = adfn::<16>::new(1.0, [2.0; 16]);
    let b  = adfn::<16>::constant(2.0);
    let mut c = adfn::<16>::constant(0.0);
    let start = Instant::now();
    for _ in 0..1000 {
         c += a.sin() * b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);
}