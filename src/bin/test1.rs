use std::time::Instant;
use vek::{Vec16, Vec4};
use ad_trait::forward_ad::adf::adf;
use ad_trait::forward_ad::adf_g::adf_g;
use ad_trait::forward_ad::adfm::{adfm, adfm2};


fn main() {
    let mut a = 1.0;
    let b = 1.0001;
    let start = Instant::now();
    for _ in 0..1000 {
        a = a * b;
    }
    println!("{:?}, {:?}", start.elapsed(), a);

    let mut a = adf::new(1.0, 1.0);
    let b = adf::new(1.0001, 1.0);

    let start = Instant::now();
    for _ in 0..1000 {
        a = a * b;
    }
    println!("{:?}, {:?}", start.elapsed(), a);

    let mut a = adf_g::new(1.0, 1.0);
    let b = adf_g::new(1.0001, 1.0);

    let start = Instant::now();
    for _ in 0..1000 {
        a = a * b;
    }
    println!("{:?}, {:?}", start.elapsed(), a);

    let mut a = adfm::new_constant(1.0);
    let b = adfm::new(1.00001, Vec4::from([1.0; 4]));

    let start = Instant::now();
    for _ in 0..1000 {
        a = a * b;
    }
    println!("{:?}, {:?}", start.elapsed(), a);

    let mut a = adfm2::new_constant(1.0);
    let b = adfm2::new(1.00001, [1.0; 16]);

    let start = Instant::now();
    for _ in 0..1000 {
        a = a * b;
    }
    println!("{:?}, {:?}", start.elapsed(), a);
}
