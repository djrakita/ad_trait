#![feature(portable_simd)]

use std::time::Instant;

fn main() {
    let a = std::simd::f32x8::from_slice(&[1.0; 64]);
    let b = std::simd::f32x8::from_slice(&[2.0; 64]);

    let start = Instant::now();
    for _ in 0..1000000 {
        let _res = a + b;
        std::hint::black_box(_res);
    }
    println!("{:?}", start.elapsed());

    ///////////

    let a = [1.; 4];
    let b = [2.; 4];

    let start = Instant::now();
    for _ in 0..1000000 {
        let mut res = [0.0; 4];
        for i in 0..4 {
            res[i] = a[i] + b[i];
        }
        std::hint::black_box(res);
    }
    println!("{:?}", start.elapsed());

    ///////////

    let a = 1.0;
    let b = 2.0;

    let start = Instant::now();
    for _ in 0..1000000 {
        let _res = a + b;
        std::hint::black_box(_res);
    }
    println!("{:?}", start.elapsed());
}