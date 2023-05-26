
// This is not finished.  You probably don't want it anyway because conditional branching may be incorrect.

/*
use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{Dim, Matrix, RawStorageMut};
use num_traits::{Bounded, Float, FloatConst, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};
use crate::{AD, ADNumType, F64};

#[allow(non_camel_case_types)]
#[derive(Clone, Debug, Copy)]
pub struct f64xn<const N: usize> {
    pub (crate) value: [f64; N]
}
impl<const N: usize> f64xn<N> {
    pub fn new(value: [f64; N]) -> Self {
        Self {
            value
        }
    }
    pub fn splat(value: f64) -> Self {
        Self {
            value: [value; N]
        }
    }
    #[inline]
    pub fn value(&self) -> [f64; N] {
        self.value
    }
}
impl<const N: usize> Index<usize> for f64xn<N> {
    type Output = f64;

    fn index(&self, index: usize) -> &Self::Output {
        return &self.value[index];
    }
}
impl<const N: usize> IndexMut<usize> for f64xn<N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.value[index]
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
impl<const N: usize> Add<F64> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F64) -> Self::Output {
        AD::add_scalar(rhs.0, self)
    }
}

impl<const N: usize> AddAssign<F64> for f64xn<N> {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Mul<F64> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F64) -> Self::Output {
        AD::mul_scalar(rhs.0, self)
    }
}

impl<const N: usize> MulAssign<F64> for f64xn<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Sub<F64> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F64) -> Self::Output {
        AD::sub_r_scalar(self, rhs.0)
    }
}

impl<const N: usize> SubAssign<F64> for f64xn<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        *self = *self - rhs;
    }
}

impl<const N: usize> Div<F64> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: F64) -> Self::Output {
        AD::div_r_scalar(self, rhs.0)
    }
}

impl<const N: usize> DivAssign<F64> for f64xn<N> {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Rem<F64> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: F64) -> Self::Output {
        AD::rem_r_scalar(self, rhs.0)
    }
}

impl<const N: usize> RemAssign<F64> for  f64xn<N> {
    #[inline]
    fn rem_assign(&mut self, rhs: F64) {
        *self = *self % rhs;
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> Add<Self> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let mut value = [0.0; N];

        for i in 0..N { value[i] = self[i] + rhs[i]; }

        Self {
            value
        }
    }
}
impl<const N: usize> AddAssign<Self> for f64xn<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Mul<Self> for f64xn<N> {
    type Output = Self;

        #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let mut value = [0.0; N];

        for i in 0..N { value[i] = self[i] * rhs[i]; }

        Self {
            value
        }
    }
}
impl<const N: usize> MulAssign<Self> for f64xn<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Sub<Self> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let mut value = [0.0; N];

        for i in 0..N { value[i] = self[i] - rhs[i]; }

        Self {
            value
        }
    }
}
impl<const N: usize> SubAssign<Self> for f64xn<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> Div<Self> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let mut value = [0.0; N];

        for i in 0..N { value[i] = self[i] / rhs[i]; }

        Self {
            value
        }
    }
}
impl<const N: usize> DivAssign<Self> for f64xn<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Rem<Self> for f64xn<N> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        todo!()
        // self - ComplexField::floor((self/rhs))*rhs
        // self - (self / rhs).floor() * rhs
    }
}
impl<const N: usize> RemAssign<Self> for f64xn<N> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl<const N: usize> Neg for f64xn<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        let mut value = [0.0; N];

        for i in 0..N { value[i] = -self[i]; }

        Self {
            value
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
impl<const N: usize> Float for f64xn<N> {
    fn nan() -> Self {
        Self::constant(f64::NAN)
    }

    fn infinity() -> Self {
        Self::constant(f64::INFINITY)
    }

    fn neg_infinity() -> Self {
        Self::constant(f64::NEG_INFINITY)
    }

    fn neg_zero() -> Self { -Self::zero() }

    fn min_value() -> Self { Self::constant(f64::MIN) }

    fn min_positive_value() -> Self {
        Self::constant(f64::MIN_POSITIVE)
    }

    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }

    fn is_nan(self) -> bool { self.value.is_nan() }

    fn is_infinite(self) -> bool {
        self.value.is_infinite()
    }

    fn is_finite(self) -> bool {
        self.value.is_finite()
    }

    fn is_normal(self) -> bool {
        self.value.is_normal()
    }

    fn classify(self) -> FpCategory {
        self.value.classify()
    }

    fn floor(self) -> Self { ComplexField::floor(self) }

    fn ceil(self) -> Self {
        ComplexField::ceil(self)
    }

    fn round(self) -> Self {
        ComplexField::round(self)
    }

    fn trunc(self) -> Self {
        ComplexField::trunc(self)
    }

    fn fract(self) -> Self {
        ComplexField::fract(self)
    }

    fn abs(self) -> Self {
        ComplexField::abs(self)
    }

    fn signum(self) -> Self {
        ComplexField::signum(self)
    }

    fn is_sign_positive(self) -> bool { RealField::is_sign_positive(&self) }

    fn is_sign_negative(self) -> bool { RealField::is_sign_negative(&self) }

    fn mul_add(self, a: Self, b: Self) -> Self { ComplexField::mul_add(self, a, b) }

    fn recip(self) -> Self { ComplexField::recip(self) }

    fn powi(self, n: i32) -> Self {
        ComplexField::powi(self, n)
    }

    fn powf(self, n: Self) -> Self {
        ComplexField::powf(self, n)
    }

    fn sqrt(self) -> Self {
        ComplexField::sqrt(self)
    }

    fn exp(self) -> Self {
        ComplexField::exp(self)
    }

    fn exp2(self) -> Self {
        ComplexField::exp2(self)
    }

    fn ln(self) -> Self {
        ComplexField::ln(self)
    }

    fn log(self, base: Self) -> Self {
        ComplexField::log(self, base)
    }

    fn log2(self) -> Self {
        ComplexField::log2(self)
    }

    fn log10(self) -> Self {
        ComplexField::log10(self)
    }

    fn max(self, other: Self) -> Self {
        RealField::max(self, other)
    }

    fn min(self, other: Self) -> Self {
        RealField::min(self, other)
    }

    fn abs_sub(self, other: Self) -> Self {
        Signed::abs_sub(&self, &other)
    }

    fn cbrt(self) -> Self { ComplexField::cbrt(self) }

    fn hypot(self, other: Self) -> Self {
        ComplexField::hypot(self, other)
    }

    fn sin(self) -> Self {
        ComplexField::sin(self)
    }

    fn cos(self) -> Self {
        ComplexField::cos(self)
    }

    fn tan(self) -> Self {
        ComplexField::tan(self)
    }

    fn asin(self) -> Self {
        ComplexField::asin(self)
    }

    fn acos(self) -> Self {
        ComplexField::acos(self)
    }

    fn atan(self) -> Self {
        ComplexField::atan(self)
    }

    fn atan2(self, other: Self) -> Self {
        RealField::atan2(self, other)
    }

    fn sin_cos(self) -> (Self, Self) {
        ComplexField::sin_cos(self)
    }

    fn exp_m1(self) -> Self {
        ComplexField::exp_m1(self)
    }

    fn ln_1p(self) -> Self {
        ComplexField::ln_1p(self)
    }

    fn sinh(self) -> Self {
        ComplexField::sinh(self)
    }

    fn cosh(self) -> Self {
        ComplexField::cosh(self)
    }

    fn tanh(self) -> Self {
        ComplexField::tanh(self)
    }

    fn asinh(self) -> Self {
        ComplexField::asinh(self)
    }

    fn acosh(self) -> Self {
        ComplexField::acosh(self)
    }

    fn atanh(self) -> Self {
        ComplexField::atanh(self)
    }

    fn integer_decode(self) -> (u64, i16, i8) { return self.value.integer_decode() }
}

impl<const N: usize> NumCast for f64xn<N> {
    fn from<T: ToPrimitive>(n: T) -> Option<Self> { unimplemented!() }
}

impl<const N: usize> ToPrimitive for f64xn<N> {
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize>  PartialEq for f64xn<N> {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<const N: usize>  PartialOrd for f64xn<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<const N: usize>  Display for f64xn<N> {
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl<const N: usize>  From<f64> for f64xn<N> {
    fn from(value: f64) -> Self {
        Self::new(value, [0.0; N])
    }
}
impl<const N: usize>  Into<f64> for f64xn<N> {
    fn into(self) -> f64 {
        self.value
    }
}
impl<const N: usize>  From<f32> for f64xn<N> {
    fn from(value: f32) -> Self {
        Self::new(value as f64, [0.0; N])
    }
}
impl<const N: usize>  Into<f32> for f64xn<N> {
    fn into(self) -> f32 {
        self.value as f32
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
*/