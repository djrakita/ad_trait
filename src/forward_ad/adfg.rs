// This is working, just uncomment everything if you ever want to include nested adfs.  I am leaving
// this out for now to simplify this crate.

/*

use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{abs, Dim, Matrix, RawStorageMut};
use num_traits::{Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use num_traits::real::Real;
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};
use crate::{AD, ADNumType, F64};

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Copy)]
pub struct adfg<const N: usize, T: AD> {
    pub (crate) value: T,
    pub (crate) tangent: [T; N]
}
impl<const N: usize, T: AD> adfg<N, T> {
    pub fn new(value: T, tangent: [T; N]) -> Self {
        Self {
            value,
            tangent
        }
    }
    pub fn new_constant(value: T) -> Self {
        Self {
            value,
            tangent: [T::zero(); N]
        }
    }
    #[inline(always)]
    pub fn value(&self) -> T {
        self.value
    }
    #[inline(always)]
    pub fn tangent(&self) -> [T; N] {
        self.tangent
    }
}

impl<const N: usize, T: AD> AD for adfg<N, T> {
    fn constant(constant: f64) -> Self {
        Self {
            value: T::constant(constant),
            tangent: [T::zero(); N]
        }
    }

    fn to_constant(&self) -> f64 {
        self.value.to_constant()
    }

    fn ad_num_type() -> ADNumType {
        ADNumType::ADF
    }

    fn add_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) + arg2
    }

    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) - arg2
    }

    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 -  Self::constant(arg2)
    }

    fn mul_scalar(arg1: f64, arg2: Self) -> Self {
         Self::constant(arg1) * arg2
    }

    fn div_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) / arg2
    }

    fn div_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 / Self::constant(arg2)
    }

    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) % arg2
    }

    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 % Self::constant(arg2)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[inline(always)]
fn two_vecs_mul_and_add<const N: usize, T: AD>(vec1: &[T; N], vec2: &[T; N], scalar1: T, scalar2: T) -> [T; N] {
    let mut out = [T::zero(); N];
    for i in 0..N {
        out[i] = scalar1*vec1[i] + scalar2*vec2[i];
    }
    out
}
#[inline(always)]
fn two_vecs_add<const N: usize, T: AD>(vec1: &[T; N], vec2: &[T; N]) -> [T; N] {
    let mut out = [T::zero(); N];
    for i in 0..N {
        out[i] = vec1[i] + vec2[i];
    }
    out
}
#[inline(always)]
fn one_vec_mul<const N: usize, T: AD>(vec: &[T; N], scalar: T) -> [T; N] {
    let mut out = [T::zero(); N];
    for i in 0..N {
        out[i] = scalar*vec[i];
    }
    out
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD> Add<F64> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F64) -> Self::Output {
        AD::add_scalar(rhs.0, self)
    }
}

impl<const N: usize, T: AD> AddAssign<F64> for adfg<N, T> {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        *self = *self + rhs;
    }
}

impl<const N: usize, T: AD> Mul<F64> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F64) -> Self::Output {
        AD::mul_scalar(rhs.0, self)
    }
}

impl<const N: usize, T: AD> MulAssign<F64> for adfg<N, T> {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        *self = *self * rhs;
    }
}

impl<const N: usize, T: AD> Sub<F64> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F64) -> Self::Output {
        AD::sub_r_scalar(self, rhs.0)
    }
}

impl<const N: usize, T: AD> SubAssign<F64> for adfg<N, T> {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        *self = *self - rhs;
    }
}

impl<const N: usize, T: AD> Div<F64> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: F64) -> Self::Output {
        AD::div_r_scalar(self, rhs.0)
    }
}

impl<const N: usize, T: AD> DivAssign<F64> for adfg<N, T> {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        *self = *self / rhs;
    }
}

impl<const N: usize, T: AD> Rem<F64> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: F64) -> Self::Output {
        AD::rem_r_scalar(self, rhs.0)
    }
}

impl<const N: usize, T: AD> RemAssign<F64> for  adfg<N, T> {
    #[inline]
    fn rem_assign(&mut self, rhs: F64) {
        *self = *self % rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD> Add<Self> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let output_value = self.value + rhs.value;
        let output_tangent = two_vecs_add(&self.tangent, &rhs.tangent);

        Self {
            value: output_value,
            tangent: output_tangent
        }

    }
}
impl<const N: usize, T: AD> AddAssign<Self> for adfg<N, T> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize, T: AD> Mul<Self> for adfg<N, T> {
    type Output = Self;

        #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let output_value = self.value * rhs.value;
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &rhs.tangent, rhs.value, self.value);

        Self {
            value: output_value,
            tangent: output_tangent
        }

    }
}
impl<const N: usize, T: AD> MulAssign<Self> for adfg<N, T> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize, T: AD> Sub<Self> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let output_value = self.value - rhs.value;
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &rhs.tangent, T::constant(1.0), T::constant(-1.0));

        Self {
            value: output_value,
            tangent: output_tangent
        }

    }
}
impl<const N: usize, T: AD> SubAssign<Self> for adfg<N, T> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize, T: AD> Div<Self> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let output_value = self.value / rhs.value;
        let d_div_d_arg1 = T::constant(1.0)/rhs.value;
        let d_div_d_arg2 = -self.value/(rhs.value*rhs.value);
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &rhs.tangent, d_div_d_arg1, d_div_d_arg2);

        Self {
            value: output_value,
            tangent: output_tangent
        }

    }
}
impl<const N: usize, T: AD> DivAssign<Self> for adfg<N, T> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize, T: AD> Rem<Self> for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self - ComplexField::floor((self/rhs))*rhs
        // self - (self / rhs).floor() * rhs
    }
}
impl<const N: usize, T: AD> RemAssign<Self> for adfg<N, T> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl<const N: usize, T: AD> Neg for adfg<N, T> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            tangent: one_vec_mul(&self.tangent, T::constant(-1.0))
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD> Float for adfg<N, T> {
    #[inline]
    fn nan() -> Self {
        Self::constant(f64::NAN)
    }

    #[inline]
    fn infinity() -> Self {
        Self::constant(f64::INFINITY)
    }

    #[inline]
    fn neg_infinity() -> Self {
        Self::constant(f64::NEG_INFINITY)
    }

    #[inline]
    fn neg_zero() -> Self { -Self::zero() }

    #[inline]
    fn min_value() -> Self { Self::constant(f64::MIN) }

    #[inline]
    fn min_positive_value() -> Self {
        Self::constant(f64::MIN_POSITIVE)
    }

    #[inline]
    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }

    #[inline]
    fn is_nan(self) -> bool { self.value.is_nan() }

    #[inline]
    fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }

    #[inline]
    fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    #[inline]
    fn is_normal(self) -> bool {
        self.value.is_normal()
    }

    #[inline]
    fn classify(self) -> FpCategory {
        self.value.classify()
    }

    #[inline]
    fn floor(self) -> Self { ComplexField::floor(self) }

    #[inline]
    fn ceil(self) -> Self {
        ComplexField::ceil(self)
    }

    #[inline]
    fn round(self) -> Self {
        ComplexField::round(self)
    }

    #[inline]
    fn trunc(self) -> Self {
        ComplexField::trunc(self)
    }

    #[inline]
    fn fract(self) -> Self {
        ComplexField::fract(self)
    }

    #[inline]
    fn abs(self) -> Self {
        ComplexField::abs(self)
    }

    #[inline]
    fn signum(self) -> Self {
        ComplexField::signum(self)
    }

    #[inline]
    fn is_sign_positive(self) -> bool { RealField::is_sign_positive(&self) }

    #[inline]
    fn is_sign_negative(self) -> bool { RealField::is_sign_negative(&self) }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self { ComplexField::mul_add(self, a, b) }

    #[inline]
    fn recip(self) -> Self { ComplexField::recip(self) }

    #[inline]
    fn powi(self, n: i32) -> Self {
        ComplexField::powi(self, n)
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        ComplexField::powf(self, n)
    }

    #[inline]
    fn sqrt(self) -> Self {
        ComplexField::sqrt(self)
    }

    #[inline]
    fn exp(self) -> Self {
        ComplexField::exp(self)
    }

    #[inline]
    fn exp2(self) -> Self {
        ComplexField::exp2(self)
    }

    #[inline]
    fn ln(self) -> Self {
        ComplexField::ln(self)
    }

    #[inline]
    fn log(self, base: Self) -> Self {
        ComplexField::log(self, base)
    }

    #[inline]
    fn log2(self) -> Self {
        ComplexField::log2(self)
    }

    #[inline]
    fn log10(self) -> Self {
        ComplexField::log10(self)
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        RealField::max(self, other)
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        RealField::min(self, other)
    }

    #[inline]
    fn abs_sub(self, other: Self) -> Self {
        Signed::abs_sub(&self, &other)
    }

    #[inline]
    fn cbrt(self) -> Self { ComplexField::cbrt(self) }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        ComplexField::hypot(self, other)
    }

    #[inline]
    fn sin(self) -> Self {
        ComplexField::sin(self)
    }

    #[inline]
    fn cos(self) -> Self {
        ComplexField::cos(self)
    }

    #[inline]
    fn tan(self) -> Self {
        ComplexField::tan(self)
    }

    #[inline]
    fn asin(self) -> Self {
        ComplexField::asin(self)
    }

    #[inline]
    fn acos(self) -> Self {
        ComplexField::acos(self)
    }

    #[inline]
    fn atan(self) -> Self {
        ComplexField::atan(self)
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        RealField::atan2(self, other)
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        ComplexField::sin_cos(self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        ComplexField::exp_m1(self)
    }

    #[inline]
    fn ln_1p(self) -> Self {
        ComplexField::ln_1p(self)
    }

    #[inline]
    fn sinh(self) -> Self {
        ComplexField::sinh(self)
    }

    #[inline]
    fn cosh(self) -> Self {
        ComplexField::cosh(self)
    }

    #[inline]
    fn tanh(self) -> Self {
        ComplexField::tanh(self)
    }

    #[inline]
    fn asinh(self) -> Self {
        ComplexField::asinh(self)
    }

    #[inline]
    fn acosh(self) -> Self {
        ComplexField::acosh(self)
    }

    #[inline]
    fn atanh(self) -> Self {
        ComplexField::atanh(self)
    }

    #[inline]
    fn integer_decode(self) -> (u64, i16, i8) { return self.value.integer_decode() }
}

impl<const N: usize, T: AD> NumCast for adfg<N, T> {
    fn from<M: ToPrimitive>(n: M) -> Option<Self> { unimplemented!() }
}

impl<const N: usize, T: AD> ToPrimitive for adfg<N, T> {
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD>  PartialEq for adfg<N, T> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<const N: usize, T: AD>  PartialOrd for adfg<N, T> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<const N: usize, T: AD>  Display for adfg<N, T> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl<const N: usize, T: AD>  From<f64> for adfg<N, T> {
    fn from(value: f64) -> Self {
        Self::new(T::constant(value), [T::zero(); N])
    }
}
impl<const N: usize, T: AD>  Into<f64> for adfg<N, T> {
    fn into(self) -> f64 {
        self.value.to_constant()
    }
}
impl<const N: usize, T: AD>  From<f32> for adfg<N, T> {
    fn from(value: f32) -> Self {
        Self::new(T::constant(value as f64), [T::zero(); N])
    }
}
impl<const N: usize, T: AD>  Into<f32> for adfg<N, T> {
    fn into(self) -> f32 {
        self.value.to_constant() as f32
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD>  UlpsEq for adfg<N, T> {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl<const N: usize, T: AD>  AbsDiffEq for adfg<N, T> {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if ComplexField::abs(diff) < epsilon {
            true
        } else {
            false
        }
    }
}

impl<const N: usize, T: AD>  RelativeEq for adfg<N, T> {
    fn default_max_relative() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, _max_relative: Self::Epsilon) -> bool {
        let diff = *self - *other;
        if ComplexField::abs(diff) < epsilon {
            true
        } else {
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD> SimdValue for adfg<N, T> {
    type Element = Self;
    type SimdBool = bool;

    fn lanes() -> usize { 4 }

    fn splat(val: Self::Element) -> Self {
        val
    }

    fn extract(&self, _: usize) -> Self::Element {
        *self
    }

    unsafe fn extract_unchecked(&self, _: usize) -> Self::Element {
        *self
    }

    fn replace(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    unsafe fn replace_unchecked(&mut self, _: usize, val: Self::Element) {
        *self = val
    }

    fn select(self, cond: Self::SimdBool, other: Self) -> Self {
        if cond {
            self
        } else {
            other
        }
    }
}

impl<const N: usize, T: AD, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adfg<N, T>, R, C>> Mul<Matrix<adfg<N, T>, R, C, S>> for adfg<N, T> {
    type Output = Matrix<Self, R, C, S>;

    fn mul(self, rhs: Matrix<Self, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

/*
impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<Matrix<f64, R, C, S>> for adfg<N, T> {
    type Output = Matrix<f64, R, C, S>;

    fn mul(self, rhs: Matrix<f64, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}
*/

impl<const N: usize, T: AD, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adfg<N, T>, R, C>> Mul<&Matrix<adfg<N, T>, R, C, S>> for adfg<N, T> {
    type Output = Matrix<Self, R, C, S>;

    fn mul(self, rhs: &Matrix<Self, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

/*
impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<&Matrix<f64, R, C, S>> for adf {
    type Output = Matrix<f64, R, C, S>;

    fn mul(self, rhs: &Matrix<f64, R, C, S>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD> Zero for adfg<N, T> {
    #[inline]
    fn zero() -> Self {
        return Self::constant(0.0)
    }

    fn is_zero(&self) -> bool {
        return self.value == T::zero();
    }
}

impl<const N: usize, T: AD> One for adfg<N, T> {
    #[inline]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl<const N: usize, T: AD> Num for adfg<N, T> {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = f64::from_str_radix(str, radix).expect("error");
        Ok(Self::constant(val))
    }
}

impl<const N: usize, T: AD> Signed for adfg<N, T> {
    #[inline]
    fn abs(&self) -> Self {
        let output_value = <T as Signed>::abs(&self.value);
        let output_tangent = if self.value.to_constant() >= 0.0 {
            self.tangent
        } else {
            one_vec_mul(&self.tangent, T::constant(-1.0))
        };

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn abs_sub(&self, other: &Self) -> Self {
        return if *self <= *other {
            Self::constant(0.0)
        } else {
            *self - *other
        };
    }

    #[inline]
    fn signum(&self) -> Self {
        let output_value = <T as Signed>::signum(&self.value);
        let output_tangent = [T::zero(); N];
        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn is_positive(&self) -> bool {
        return self.value.to_constant() > 0.0;
    }

    #[inline]
    fn is_negative(&self) -> bool {
        return self.value.to_constant() < 0.0;
    }
}

impl<const N: usize, T: AD> FromPrimitive for adfg<N, T> {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }
}

impl<const N: usize, T: AD> Bounded for adfg<N, T> {
    #[inline]
    fn min_value() -> Self {
        Self::constant(f64::MIN)
    }

    #[inline]
    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize, T: AD> RealField for adfg<N, T> {
    #[inline]
    fn is_sign_positive(&self) -> bool {
        return self.is_positive();
    }

    #[inline]
    fn is_sign_negative(&self) -> bool {
        return self.is_negative();
    }

    #[inline]
    fn copysign(self, sign: Self) -> Self {
        return if sign.is_positive() {
            ComplexField::abs(self)
        } else {
            -ComplexField::abs(self)
        };
    }

    #[inline]
    fn max(self, other: Self) -> Self {
        let output_value = self.value.max(other.value);
        let output_tangent = if self >= other {
            self.tangent
        } else {
            other.tangent
        };
        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let output_value = self.value.min(other.value);
        let output_tangent = if self <= other {
            self.tangent
        } else {
            other.tangent
        };
        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min <= max);
        return RealField::min(RealField::max(self, min), max);
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let output_value = self.value.atan2(other.value);
        let d_atan2_d_arg1 = other.value/(self.value*self.value + other.value*other.value);
        let d_atan2_d_arg2 = -self.value/(self.value*self.value + other.value*other.value);
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &other.tangent, d_atan2_d_arg1, d_atan2_d_arg2);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn min_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }

    #[inline]
    fn max_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }

    #[inline]
    fn pi() -> Self {
        Self::constant(std::f64::consts::PI)
    }

    #[inline]
    fn two_pi() -> Self {
        Self::constant(2.0 * std::f64::consts::PI)
    }

    #[inline]
    fn frac_pi_2() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_2)
    }

    #[inline]
    fn frac_pi_3() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_3)
    }

    #[inline]
    fn frac_pi_4() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_4)
    }

    #[inline]
    fn frac_pi_6() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_6)
    }

    #[inline]
    fn frac_pi_8() -> Self {
        Self::constant(std::f64::consts::FRAC_PI_8)
    }

    #[inline]
    fn frac_1_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_1_PI)
    }

    #[inline]
    fn frac_2_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_2_PI)
    }

    #[inline]
    fn frac_2_sqrt_pi() -> Self {
        Self::constant(std::f64::consts::FRAC_2_SQRT_PI)
    }

    #[inline]
    fn e() -> Self {
        Self::constant(std::f64::consts::E)
    }

    #[inline]
    fn log2_e() -> Self {
        Self::constant(std::f64::consts::LOG2_E)
    }

    #[inline]
    fn log10_e() -> Self {
        Self::constant(std::f64::consts::LOG10_E)
    }

    #[inline]
    fn ln_2() -> Self {
        Self::constant(std::f64::consts::LN_2)
    }

    #[inline]
    fn ln_10() -> Self {
        Self::constant(std::f64::consts::LN_10)
    }
}

impl<const N: usize, T: AD> ComplexField for adfg<N, T> {
    type RealField = Self;

    #[inline]
    fn from_real(re: Self::RealField) -> Self { re.clone() }

    #[inline]
    fn real(self) -> <Self as ComplexField>::RealField { self.clone() }

    #[inline]
    fn imaginary(self) -> Self::RealField { Self::zero() }

    #[inline]
    fn modulus(self) -> Self::RealField { return ComplexField::abs(self); }

    #[inline]
    fn modulus_squared(self) -> Self::RealField { self * self }

    #[inline]
    fn argument(self) -> Self::RealField { unimplemented!(); }

    #[inline]
    fn norm1(self) -> Self::RealField { return ComplexField::abs(self); }

    #[inline]
    fn scale(self, factor: Self::RealField) -> Self { return self * factor; }

    #[inline]
    fn unscale(self, factor: Self::RealField) -> Self { return self / factor; }

    #[inline]
    fn floor(self) -> Self {
        Self::new(ComplexField::floor(self.value), [T::zero(); N])
    }

    #[inline]
    fn ceil(self) -> Self {
        Self::new(ComplexField::ceil(self.value), [T::zero(); N])
    }

    #[inline]
    fn round(self) -> Self {
        Self::new(ComplexField::round(self.value), [T::zero(); N])
    }

    #[inline]
    fn trunc(self) -> Self {
        Self::new(ComplexField::trunc(self.value), [T::zero(); N])
    }

    #[inline]
    fn fract(self) -> Self {
        Self::new(ComplexField::fract(self.value), [T::zero(); N])
    }

    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        return (self * a) + b;
    }

    #[inline]
    fn abs(self) -> Self::RealField {
        <Self as Signed>::abs(&self)
    }

    #[inline]
    fn hypot(self, other: Self) -> Self::RealField {
        return ComplexField::sqrt((ComplexField::powi(self, 2) + ComplexField::powi(other, 2)));
    }

    #[inline]
    fn recip(self) -> Self {
        return Self::constant(1.0) / self;
    }

    #[inline]
    fn conjugate(self) -> Self { return self; }

    #[inline]
    fn sin(self) -> Self {
        let output_value = ComplexField::sin(self.value);
        let d_sin_d_arg1 = ComplexField::cos(self.value);
        let output_tangent = one_vec_mul(&self.tangent, d_sin_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn cos(self) -> Self {
        let output_value = ComplexField::cos(self.value);
        let d_cos_d_arg1 =  -ComplexField::sin(self.value);
        let output_tangent = one_vec_mul(&self.tangent, d_cos_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        return (ComplexField::sin(self), ComplexField::cos(self));
    }

    #[inline]
    fn tan(self) -> Self {
        let output_value = ComplexField::tan(self.value);
        let c = ComplexField::cos(self.value);
        let d_tan_d_arg1 =  T::one()/(c*c);
        let output_tangent = one_vec_mul(&self.tangent, d_tan_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn asin(self) -> Self {
        let output_value = ComplexField::asin(self.value);
        let d_asin_d_arg1 =  T::one() / ComplexField::sqrt((T::one() - self.value * self.value));
        let output_tangent = one_vec_mul(&self.tangent, d_asin_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn acos(self) -> Self {
        let output_value = ComplexField::acos(self.value);
        let d_acos_d_arg1 =  T::constant(-1.0) / ComplexField::sqrt((T::constant(1.0) - self.value * self.value));
        let output_tangent = one_vec_mul(&self.tangent, d_acos_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn atan(self) -> Self {
        let output_value = ComplexField::atan(self.value);
        let d_atan_d_arg1 =  T::constant(1.0) / (self.value * self.value + T::constant(1.0));
        let output_tangent = one_vec_mul(&self.tangent, d_atan_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn sinh(self) -> Self {
        let output_value = ComplexField::sinh(self.value);
        let d_sinh_d_arg1 =  ComplexField::cosh(self.value);
        let output_tangent = one_vec_mul(&self.tangent, d_sinh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn cosh(self) -> Self {
        let output_value = ComplexField::cosh(self.value);
        let d_cosh_d_arg1 =  ComplexField::sinh(self.value);
        let output_tangent = one_vec_mul(&self.tangent, d_cosh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn tanh(self) -> Self {
        let output_value = ComplexField::tanh(self.value);
        let c = ComplexField::cosh(self.value);
        let d_tanh_d_arg1 =  T::constant(1.0)/(c*c);
        let output_tangent = one_vec_mul(&self.tangent, d_tanh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn asinh(self) -> Self {
        let output_value = ComplexField::asinh(self.value);
        let d_asinh_d_arg1 =  T::constant(1.0)/ComplexField::sqrt((self.value*self.value + T::constant(1.0)));
        let output_tangent = one_vec_mul(&self.tangent, d_asinh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        let output_value = ComplexField::acosh(self.value);
        let d_acosh_d_arg1 =  T::constant(1.0)/(ComplexField::sqrt((self.value - T::constant(1.0)))*ComplexField::sqrt((self.value + T::constant(1.0))));
        let output_tangent = one_vec_mul(&self.tangent, d_acosh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn atanh(self) -> Self {
        let output_value = ComplexField::atanh(self.value);
        let d_atanh_d_arg1 =  T::constant(1.0)/(T::constant(1.0) - self.value*self.value);
        let output_tangent = one_vec_mul(&self.tangent, d_atanh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        let output_value = <T as Float>::log(self.value, base.value);
        let ln_rhs = ComplexField::ln(base.value);
        let ln_lhs = ComplexField::ln(self.value);
        let d_log_d_arg1 = T::constant(1.0)/(self.value * ln_rhs);
        let d_log_d_arg2 = -ln_lhs / (base.value * ln_rhs * ln_rhs);
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &base.tangent, d_log_d_arg1, d_log_d_arg2);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn log2(self) -> Self { return ComplexField::log(self, Self::constant(2.0)); }

    #[inline]
    fn log10(self) -> Self { return ComplexField::log(self, Self::constant(10.0)); }

    #[inline]
    fn ln(self) -> Self { return ComplexField::log(self, Self::constant(std::f64::consts::E)); }

    #[inline]
    fn ln_1p(self) -> Self { ComplexField::ln(Self::constant(1.0) + self) }

    #[inline]
    fn sqrt(self) -> Self {
        let output_value = ComplexField::sqrt(self.value);
        let d_sqrt_d_arg1 =  T::constant(1.0)/(T::constant(2.0)*ComplexField::sqrt(self.value));
        let output_tangent = one_vec_mul(&self.tangent, d_sqrt_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn exp(self) -> Self {
        let output_value = ComplexField::exp(self.value);
        let output_tangent = one_vec_mul(&self.tangent, output_value);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn exp2(self) -> Self { ComplexField::powf(Self::constant(2.0), self) }

    #[inline]
    fn exp_m1(self) -> Self { return ComplexField::exp(self) - Self::constant(1.0); }

    #[inline]
    fn powi(self, n: i32) -> Self { return ComplexField::powf(self, Self::constant(n as f64)); }

    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        let output_value = <T as Float>::powf(self.value, n.value);
        let d_powf_d_arg1 = n.value * <T as Float>::powf(self.value, n.value - T::constant(1.0));
        let d_powf_d_arg2 = <T as Float>::powf(self.value, n.value) * ComplexField::ln(self.value);
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &n.tangent, d_powf_d_arg1, d_powf_d_arg2);

        Self {
            value: output_value,
            tangent: output_tangent
        }
    }

    #[inline]
    fn powc(self, n: Self) -> Self { return ComplexField::powf(self, n); }

    #[inline]
    fn cbrt(self) -> Self { return ComplexField::powf(self, Self::constant(1.0 / 3.0)); }

    #[inline]
    fn is_finite(&self) -> bool { return self.value.is_finite(); }

    #[inline]
    fn try_sqrt(self) -> Option<Self> {
        Some(ComplexField::sqrt(self))
    }
}

impl<const N: usize, T: AD> SubsetOf<Self> for adfg<N, T> {
    fn to_superset(&self) -> Self {
        self.clone()
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &adfg<N, T>) -> bool {
        true
    }
}

impl<const N: usize, T: AD> Field for adfg<N, T> { }

impl<const N: usize, T: AD> PrimitiveSimdValue for adfg<N, T> {}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for f32 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as f32
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for f64 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant()
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for u32 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as u32
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for u64 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as u64
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for u128 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as u128
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for i32 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as i32
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for i64 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as i64
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

impl<const N: usize, T: AD> SubsetOf<adfg<N, T>> for i128 {
    fn to_superset(&self) -> adfg<N, T> {
        adfg::new_constant(T::constant(*self as f64))
    }

    fn from_superset_unchecked(element: &adfg<N, T>) -> Self {
        element.value.to_constant() as i128
    }

    fn is_in_subset(_: &adfg<N, T>) -> bool {
        false
    }
}

*/
