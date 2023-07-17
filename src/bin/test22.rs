/*
use std::time::Instant;
use nalgebra::DVector;
use nalgebra_sparse::{CooMatrix, CscMatrix, CsrMatrix};

fn main() {

    let mut coo = CooMatrix::<f64>::new(5000, 1000);
    // coo.push(0, 0, 1.0);
    let csr = CsrMatrix::from(&coo);
    let v = DVector::from_vec(vec![1.0; 1000]);

    let start = Instant::now();
    let res = &csr*&v;
    println!("{:?}", start.elapsed());

    let mut coo = CooMatrix::<f64>::new(50000, 10000);
    coo.push(0, 0, 1.0);
    coo.push(0, 3, 2.0);
    coo.push(1000, 800, 5.0);
    coo.push(500, 200, 6.0);
    let start = Instant::now();
    let csc = CscMatrix::from(&coo);
    println!("{:?}", start.elapsed());
    let v = DVector::from_vec(vec![7.0; 10000]);

    let start = Instant::now();
    for _ in 0..1000 {
        &csc*&v;
    }
    let res = &csc*&v;
    println!("{:?}", start.elapsed());
    // println!("{}", res);
}
*/

fn main() {}