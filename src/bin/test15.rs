
use std::time::Instant;
use simba::simd::{f32x16, f64x8, SimdValue};
use ad_trait::AD;
use ad_trait::forward_ad::adf3::{adf_f32x16, adf_f64x8};
use ad_trait::forward_ad::adf::adf;

fn main() {
    let a = 1.0;
    let b = 2.0;
    let mut c = 1000.0;
    let start = Instant::now();
    for _ in 0..1000 {
        c /= a/b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);

    let a = adf_f64x8::new(1.0, f64x8::splat(2.0));
    let b  = adf_f64x8::constant(2.0);
    let mut c = adf_f64x8::constant(1000.0);
    let start = Instant::now();
    for _ in 0..1000 {
        c /= a/b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);

    let a = adf::<16>::new(1.0, [2.0; 16]);
    let b  = adf::<16>::constant(2.0);
    let mut c = adf::<16>::constant(1000.0);
    let start = Instant::now();
    for _ in 0..1000 {
        c /= a/b;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", c);
}