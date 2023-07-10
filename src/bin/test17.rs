#![feature(generic_associated_types)]

use std::time::Instant;
use nalgebra::{DMatrix};






fn main() {

    // let s = SpiderData::new(5, 2, 1,-1.0, 1.0);

    /*
    let s = SpiderForwardAD::<Test, adfn<2>>::new(&(), 0.9999999999);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    let res = s.derivative(&[1.,2.,3.,4.], &());
    println!("{:?}", res);
    s.spider_data().print_w();
    */

    // let w = DVector::from_vec(vec![0.25, 0.25, 0.25, 0.25]);
    // let wpinv = w.clone().pseudo_inverse(0.0).unwrap();
    // println!("{}", w*wpinv);

    // let w = DMatrix::<f64>::from_partial_diagonal(2, 2, &[1.2,5.]);
    // let wp = w.clone().pseudo_inverse(0.0).unwrap();

    // println!("{}", wp * w);


    let n = 100;
    let m = DMatrix::<f64>::new_random(n, n);

    let start = Instant::now();
    let _res = m.try_inverse().unwrap();
    println!("{:?}", start.elapsed());
}