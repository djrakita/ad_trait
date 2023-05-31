use std::time::Instant;
use nalgebra::UnitQuaternion;
use simba::simd::{f32x16, f32x2, f32x4, f32x8, f64x8, SimdComplexField, SimdValue};
use ad_trait::forward_ad::adf::adf;
use ad_trait::AD;
use ad_trait::forward_ad::adf2::adf2;
use ad_trait::forward_ad::adf3::{adf_f32x16, adf_f32x8, adf_f64x8};
use ad_trait::simd::f64xn::f64xn;

fn main() {
    let q1 = UnitQuaternion::from_euler_angles(1.,2.,3.);
    let q2 = UnitQuaternion::from_euler_angles(4.,5.,6.);
    let mut q3 = UnitQuaternion::from_euler_angles(7.,8.,9.);

    let start = Instant::now();
    for _ in 0..1000 {
        q3 *= q1 * q2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", q3);


    const N: usize = 8;
    let q1 = UnitQuaternion::from_euler_angles(f64xn::<N>::splat(1.0),f64xn::<N>::splat(2.0),f64xn::<N>::splat(3.0));
    let q2 = UnitQuaternion::from_euler_angles(f64xn::<N>::splat(4.0),f64xn::<N>::splat(5.0),f64xn::<N>::splat(6.0));
    let mut q3 = UnitQuaternion::from_euler_angles(f64xn::<N>::splat(7.0),f64xn::<N>::splat(8.0),f64xn::<N>::splat(9.0));

    let start = Instant::now();
    for _ in 0..1000 {
        q3 *= q1 * q2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", q3);

    let q1 = UnitQuaternion::from_euler_angles(adf::<N>::constant(1.0),adf::<N>::constant(2.0),adf::<N>::constant(3.0));
    let q2 = UnitQuaternion::from_euler_angles(adf::<N>::constant(4.0),adf::<N>::constant(5.0),adf::<N>::constant(6.0));
    let mut q3 = UnitQuaternion::from_euler_angles(adf::<N>::constant(7.0),adf::<N>::constant(8.0),adf::<N>::constant(9.0));

    let start = Instant::now();
    for _ in 0..1000 {
        q3 *= q1 * q2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", q3);

    let q1 = UnitQuaternion::from_euler_angles(f32x2::new(1.0, 1.001),f32x2::new(2.0, 2.0002),f32x2::splat(3.0));
    let q2 = UnitQuaternion::from_euler_angles(f32x2::splat(4.0),f32x2::splat(5.0),f32x2::splat(6.0));
    let mut q3 = UnitQuaternion::from_euler_angles(f32x2::splat(7.0),f32x2::splat(8.0),f32x2::splat(9.0));

    let start = Instant::now();
    for _ in 0..1000 {
        q3 *= q1 * q2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", q3);

    let q1 = UnitQuaternion::from_euler_angles(adf_f64x8::constant(1.0),adf_f64x8::constant(2.0),adf_f64x8::constant(3.0));
    let q2 = UnitQuaternion::from_euler_angles(adf_f64x8::constant(4.0), adf_f64x8::constant(5.0),adf_f64x8::constant(6.0));
    let mut q3 = UnitQuaternion::from_euler_angles(adf_f64x8::constant(7.0),adf_f64x8::constant(8.0),adf_f64x8::constant(9.0));

    let start = Instant::now();
    for _ in 0..1000 {
        q3 *= q1 * q2;
    }
    println!("{:?}", start.elapsed());
    println!("{:?}", q3);
}