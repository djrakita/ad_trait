use nalgebra::{DVector, Vector3};
use ad_trait::AD;

pub fn test<F: AD>(a: F, b: F) -> F {
    a * b
}

pub fn test2<F: AD>(a: DVector<F>, b: DVector<F>) -> DVector<F> {
    return F::scalar_multiply_by_nalgebra_matrix(3.0, &a) + b;
}

fn main() {
    let res = test2::<f64>(DVector::from_vec(vec![1.,2.]), DVector::from_vec(vec![3.,4.]));
    println!("{:?}", res);

    let res = test(1., 2.);
    println!("{:?}", res);

    let v = Vector3::new(1.,2.,3.);
    let vv = f64::scalar_multiply_by_nalgebra_matrix(3.0, &v);
    println!("{:?}", vv);
}
