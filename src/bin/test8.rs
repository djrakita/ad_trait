use std::time::Instant;
use simba::scalar::ComplexField;
use ad_trait::AD;
use ad_trait::forward_ad::adf::adf;
use ad_trait::reverse_ad::adr::adr;
use ad_trait::simd::f64xn::f64xn;

const N: usize = 16;

fn main() {
    let a1 = adf::<N>::constant(1.0);
    let a2 = adf::<N>::constant(3.0);
    let mut a3 = adf::<N>::new(2.0, [1.0; N]);

    let start = Instant::now();
    for _ in 0..1000 {
        a3 += a1.powf(a2).sin() * a3;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", a3);

    let a1 = adr::constant(1.0);
    let a2 = adr::constant(3.0);
    let mut a3 = adr::new_variable(2.0, true);

    let start = Instant::now();
    for _ in 0..1000 {
        a3 += a1.powf(a2).sin() * a3;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", a3);

    let a1 = 1.0;
    let a2 = 3.0;
    let mut a3 = 2.0;

    let start = Instant::now();
    for _ in 0..1000 {
        a3 += a1.powf(a2).sin() * a3;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", a3);

    let a1 = f64xn::<N>::splat(1.0);
    let a2 = f64xn::<N>::splat(3.0);
    let mut a3 = f64xn::<N>::new([1.,2.,3.,4.,5.,6.,7.,8.,9.,10.,11.,12.,13.,14.,15.,16.]);

    let start = Instant::now();
    for _ in 0..1000 {
        a3 += a1.powf(a2).sin() * a3;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", a3);
}