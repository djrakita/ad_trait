use nalgebra::DMatrix;
use ad_trait::AD;
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::adr;
use ad_trait::simd::f64xn::f64xn;

fn test<T: AD>(a: T, b: T) -> T {
    return a + b;
}

struct A<T: AD> {
    m: DMatrix<T>
}
impl<T: AD> A<T> {
    pub fn new() -> Self {
        Self {
            m: DMatrix::from_vec(2, 2, vec![T::constant(3.0); 4])
        }
    }
}

fn main() {
    let a1 = A::<adfn<3>>::new();
    let a2 = A::<adfn<3>>::new();

    println!("{}", a1.m * a2.m);

    let a1 = A::<adr>::new();
    let a2 = A::<adr>::new();

    println!("{}", a1.m * a2.m);

    let a1 = A::<f32>::new();
    let a2 = A::<f32>::new();

    println!("{}", a1.m * a2.m);

    let a1 = A::<f64>::new();
    let a2 = A::<f64>::new();

    println!("{}", a1.m * a2.m);

    let a1 = A::<f64xn<16>>::new();
    let a2 = A::<f64xn<16>>::new();

    println!("{}", a1.m * a2.m);
}