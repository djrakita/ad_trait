use nalgebra::Point3;
use ad_trait::NalgebraMatMulAD;
use ad_trait::simd::f64xn::f64xn;

fn main() {
    let point = Point3::from_slice(&[f64xn::new([1.,2.,]), f64xn::new([1.,6.,]), f64xn::new([8.,2.,])]);

    // Convert the Point to a Vector (which is essentially a Matrix in nalgebra context)
    let v = &point.coords;

    let n = f64xn::new([2.,5.,]).mul_by_nalgebra_matrix_ref(v);
    println!("{:?}", n);
}