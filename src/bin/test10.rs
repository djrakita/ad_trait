use faer_core::{Mat, Parallelism};
use faer_svd::*;
use dyn_stack::*;

fn main() {
    let a = vec![1.,2.,3.,4.,5.,6.,7.,8.,9.];
    let m = Mat::with_dims(3, 3, |i, j| a[3*i + j]);

    let mut u: Mat<f64> = Mat::zeros(3,3);
    let mut s: Mat<f64> = Mat::zeros(3,1);
    let mut v: Mat<f64> = Mat::zeros(3,3);

    let mut mem = GlobalMemBuffer::new(
    compute_svd_req::<f64>(
        3,
        3,
        ComputeVectors::Full,
        ComputeVectors::Full,
        Parallelism::None,
        SvdParams::default(),
    )
    .unwrap(),
    );
    let mut stack = DynStack::new(&mut mem);

    compute_svd(m.as_ref(), s.as_mut(), Some(u.as_mut()), Some(v.as_mut()), f64::EPSILON, f64::MIN_POSITIVE, Parallelism::None, stack, SvdParams::default());

    let ss = Mat::with_dims(3, 3, |i, j| if i == j { s.read(i, 0) } else { 0.0 });
    println!("{:?}", u*ss*v.transpose());
}