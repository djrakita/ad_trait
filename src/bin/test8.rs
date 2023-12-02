use num_traits::{Float, Zero};
use simba::scalar::ComplexField;
use simba::simd::{f32x2, SimdValue};
use ad_trait::forward_ad::adf::{adf_f32x2, adf_f32x8};

#[inline(always)]
fn mul_with_nan_check(a: f32, b: f32) -> f32 {
    return if a.is_nan() && b.is_zero() { 0.0 } else if a.is_zero() && b.is_nan() { 0.0 } else { a * b }
}

fn two_vecs_mul_and_add_with_nan_check(vec1: &f32x2, vec2: &f32x2, scalar1: f32, scalar2: f32) -> f32x2 {
    let mut out_vec = vec![];

    for i in 0..2 {
        out_vec.push( mul_with_nan_check(vec1.extract(i), scalar1) + mul_with_nan_check(vec2.extract(i), scalar2) );
    }

    f32x2::from_slice_unaligned(&out_vec)
}

fn main() {
    let a = adf_f32x2::new(3.0, f32x2::new(1., 2.));
    let b = adf_f32x2::new(2.0, f32x2::new(1., f32::nan()));

    let c = two_vecs_mul_and_add_with_nan_check(&a.tangent(), &b.tangent(), 2.0, 0.0);
    println!("{:?}", c);
}