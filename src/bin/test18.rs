use std::time::Instant;
use nalgebra::{DMatrix, LU, SVD};

fn main() {
    let n = 20;

    let m = DMatrix::<f64>::new_random(n, n);
    let r = DMatrix::<f64>::new_random(n, n);

    let start = Instant::now();
    let lu = LU::new(m);
    let res = lu.solve(&r);
    println!("{:?}", start.elapsed());

    let m = DMatrix::<f64>::new_random(n, n);
    let r = DMatrix::<f64>::new_random(n, n);

    let start = Instant::now();
    let res = m.try_inverse().unwrap();
    println!("{:?}", start.elapsed());

    let m = DMatrix::<f64>::new_random(n, n);
    let r = DMatrix::<f64>::new_random(n, n);

    let start = Instant::now();
    let svd = SVD::new(m, false, false);
    let res = svd.solve(&r, 0.0);
    println!("{:?}", start.elapsed());


    let m = 1;
    let n = 500;
    let k = 502;

    let f = DMatrix::<f64>::new_random(m, k);
    let w = DMatrix::<f64>::new_random(k, n);

    let start = Instant::now();
    for _ in 0..1000 {
        &f*&w;
    }
    println!("{:?}", start.elapsed());
}