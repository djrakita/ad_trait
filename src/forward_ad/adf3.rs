use std::cmp::Ordering;
use std::fmt::{Display, Formatter};
use std::num::FpCategory;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use faer_core::Entity;
use faer_core::pulp::Simd;
use nalgebra::{Dim, Matrix, RawStorageMut};
use num_traits::{Bounded, Float, FromPrimitive, Num, NumCast, One, Signed, ToPrimitive, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{f32x16, f32x4, f32x8, f32x2, f64x2, f64x4, f64x8, PrimitiveSimdValue, SimdValue};
use crate::{AD, F64};

#[macro_export]
macro_rules! make_adf {
    ($t: tt, $v: tt, $s: tt, $a: tt) => {
        #[allow(non_camel_case_types)]
        #[derive(Clone, Debug, Copy)]
        pub struct $s {
            pub (crate) value: f64,
            pub (crate) tangent: $t
        }
        impl $s {
            pub fn new(value: f64, tangent: $t) -> Self {
                Self {
                    value,
                    tangent
                }
            }
            pub fn constant(value: f64) -> Self {
                Self {
                    value,
                    tangent: $t::zero()
                }
            }
            #[inline]
            pub fn value(&self) -> f64 {
                self.value
            }
            #[inline]
            pub fn tangent(&self) -> $t {
                self.tangent
            }
            pub fn tangent_size() -> usize {
                $a
            }
        }

        impl AD for $s {
            fn constant(constant: f64) -> Self {
                Self {
                    value: constant,
                    tangent: $t::zero()
                }
            }

            fn to_constant(&self) -> f64 {
                self.value
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

        impl Add<F64> for $s {
            type Output = Self;

            #[inline]
            fn add(self, rhs: F64) -> Self::Output {
                AD::add_scalar(rhs.0, self)
            }
        }

        impl AddAssign<F64> for $s {
            #[inline]
            fn add_assign(&mut self, rhs: F64) {
                *self = *self + rhs;
            }
        }

        impl Mul<F64> for $s {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: F64) -> Self::Output {
                AD::mul_scalar(rhs.0, self)
            }
        }

        impl MulAssign<F64> for $s {
            #[inline]
            fn mul_assign(&mut self, rhs: F64) {
                *self = *self * rhs;
            }
        }

        impl Sub<F64> for $s {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: F64) -> Self::Output {
                AD::sub_r_scalar(self, rhs.0)
            }
        }

        impl SubAssign<F64> for $s {
            #[inline]
            fn sub_assign(&mut self, rhs: F64) {
                *self = *self - rhs;
            }
        }

        impl Div<F64> for $s {
            type Output = Self;

            #[inline]
            fn div(self, rhs: F64) -> Self::Output {
                AD::div_r_scalar(self, rhs.0)
            }
        }

        impl DivAssign<F64> for $s {
            #[inline]
            fn div_assign(&mut self, rhs: F64) {
                *self = *self / rhs;
            }
        }

        impl Rem<F64> for $s {
            type Output = Self;

            #[inline]
            fn rem(self, rhs: F64) -> Self::Output {
                AD::rem_r_scalar(self, rhs.0)
            }
        }

        impl RemAssign<F64> for  $s {
            #[inline]
            fn rem_assign(&mut self, rhs: F64) {
                *self = *self % rhs;
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        impl Add<Self> for $s {
            type Output = Self;

            #[inline]
            fn add(self, rhs: Self) -> Self::Output {
                let output_value = self.value + rhs.value;
                let output_tangent = self.tangent + rhs.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }

            }
        }
        impl AddAssign<Self> for $s {
            #[inline]
            fn add_assign(&mut self, rhs: Self) {
                *self = *self + rhs;
            }
        }

        impl Mul<Self> for $s {
            type Output = Self;

            #[inline]
            fn mul(self, rhs: Self) -> Self::Output {
                let output_value = self.value * rhs.value;
                let output_tangent = $t::splat(rhs.value as $v)*self.tangent + $t::splat(self.value as $v)*rhs.tangent;
                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }
        }
        impl MulAssign<Self> for $s {
            #[inline]
            fn mul_assign(&mut self, rhs: Self) {
                *self = *self * rhs;
            }
        }

        impl Sub<Self> for $s {
            type Output = Self;

            #[inline]
            fn sub(self, rhs: Self) -> Self::Output {
                let output_value = self.value - rhs.value;
                let output_tangent = self.tangent + -$t::one()*rhs.tangent;
                Self {
                    value: output_value,
                    tangent: output_tangent
                }

            }
        }
        impl SubAssign<Self> for $s {
            #[inline]
            fn sub_assign(&mut self, rhs: Self) {
                *self = *self - rhs;
            }
        }

        impl Div<Self> for $s {
            type Output = Self;

            #[inline]
            fn div(self, rhs: Self) -> Self::Output {
                let output_value = self.value / rhs.value;
                let d_div_d_arg1 = (1.0/rhs.value) as $v;
                let d_div_d_arg2 = (-self.value/(rhs.value*rhs.value)) as $v;
                let output_tangent = $t::splat(d_div_d_arg1)*self.tangent + $t::splat(d_div_d_arg2)*rhs.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }

            }
        }
        impl DivAssign<Self> for $s {
            #[inline]
            fn div_assign(&mut self, rhs: Self) {
                *self = *self / rhs;
            }
        }

        impl Rem<Self> for $s {
            type Output = Self;

            #[inline]
            fn rem(self, rhs: Self) -> Self::Output {
                todo!()
                // self - ComplexField::floor(self/rhs)*rhs
                // self - (self / rhs).floor() * rhs
            }
        }
        impl RemAssign<Self> for $s {
            #[inline]
            fn rem_assign(&mut self, rhs: Self) {
                *self = *self % rhs;
            }
        }

        impl Neg for $s {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self::Output {
                Self {
                    value: -self.value,
                    tangent: -$t::one()*self.tangent
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        /*
        impl Float for $s {
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

        impl NumCast for $s {
            fn from<T: ToPrimitive>(_n: T) -> Option<Self> { unimplemented!() }
        }

        impl ToPrimitive for $s {
            fn to_i64(&self) -> Option<i64> {
                self.value.to_i64()
            }

            fn to_u64(&self) -> Option<u64> {
                self.value.to_u64()
            }
        }
        */

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        impl  PartialEq for $s {
            #[inline]
            fn eq(&self, other: &Self) -> bool {
                self.value == other.value
            }
        }

        impl  PartialOrd for $s {
            #[inline]
            fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
                self.value.partial_cmp(&other.value)
            }
        }

        impl  Display for $s {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                f.write_str(&format!("{:?}", self)).expect("error");
                Ok(())
            }
        }

        impl  From<f64> for $s {
            fn from(value: f64) -> Self {
                Self::new(value, $t::splat(0.0))
            }
        }
        impl  Into<f64> for $s {
            fn into(self) -> f64 {
                self.value
            }
        }
        impl  From<f32> for $s {
            fn from(value: f32) -> Self {
                Self::new(value as f64, $t::splat(0.0))
            }
        }
        impl  Into<f32> for $s {
            fn into(self) -> f32 {
                self.value as f32
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        impl  UlpsEq for $s {
            fn default_max_ulps() -> u32 {
                unimplemented!("take the time to figure this out.")
            }

            fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
                unimplemented!("take the time to figure this out.")
            }
        }

        impl  AbsDiffEq for $s {
            type Epsilon = Self;

            fn default_epsilon() -> Self::Epsilon {
                Self::constant(0.000000001)
            }

            fn abs_diff_eq(&self, other: &Self, epsilon: Self::Epsilon) -> bool {
                let diff = *self - *other;
                if diff.value < epsilon.value {
                    true
                } else {
                    false
                }
            }
        }

        impl  RelativeEq for $s {
            fn default_max_relative() -> Self::Epsilon {
                Self::constant(0.000000001)
            }

            fn relative_eq(&self, other: &Self, epsilon: Self::Epsilon, _max_relative: Self::Epsilon) -> bool {
                let diff = *self - *other;
                if diff.value < epsilon.value {
                    true
                } else {
                    false
                }
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        impl SimdValue for $s {
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

        impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<$s, R, C>> Mul<Matrix<$s, R, C, S>> for $s {
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
        impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<Matrix<f64, R, C, S>> for $s {
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

        impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<$s, R, C>> Mul<&Matrix<$s, R, C, S>> for $s {
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

        impl Zero for $s {
            #[inline(always)]
            fn zero() -> Self {
                return Self::constant(0.0)
            }

            fn is_zero(&self) -> bool {
                return self.value == 0.0;
            }
        }

        impl One for $s {
            #[inline(always)]
            fn one() -> Self {
                Self::constant(1.0)
            }
        }

        impl Num for $s {
            type FromStrRadixErr = ();

            fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
                let val = f64::from_str_radix(str, radix).expect("error");
                Ok(Self::constant(val))
            }
        }

        impl Signed for $s {

            #[inline]
            fn abs(&self) -> Self {
                let output_value = self.value.abs();
                let output_tangent = if self.value >= 0.0 {
                    self.tangent
                } else {
                    // one_vec_mul(&self.tangent, -1.0)
                    -$t::one()*self.tangent
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
                let output_value = self.value.signum();
                let output_tangent = $t::zero();
                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            fn is_positive(&self) -> bool {
                return self.value > 0.0;
            }

            fn is_negative(&self) -> bool {
                return self.value < 0.0;
            }
        }

        impl FromPrimitive for $s {
            fn from_i64(n: i64) -> Option<Self> {
                Some(Self::constant(n as f64))
            }

            fn from_u64(n: u64) -> Option<Self> {
                Some(Self::constant(n as f64))
            }
        }

        impl Bounded for $s {
            fn min_value() -> Self {
                Self::constant(f64::MIN)
            }

            fn max_value() -> Self {
                Self::constant(f64::MAX)
            }
        }

        ////////////////////////////////////////////////////////////////////////////////////////////////////

        impl RealField for $s {
            fn is_sign_positive(&self) -> bool {
                return self.is_positive();
            }

            fn is_sign_negative(&self) -> bool {
                return self.is_negative();
            }

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
                let d_atan2_d_arg1 = (other.value/(self.value*self.value + other.value*other.value)) as $v;
                let d_atan2_d_arg2 = (-self.value/(self.value*self.value + other.value*other.value)) as $v;
                let output_tangent = $t::splat(d_atan2_d_arg1)*self.tangent + $t::splat(d_atan2_d_arg2)*other.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            fn min_value() -> Option<Self> {
                Some(Self::constant(f64::MIN))
            }

            fn max_value() -> Option<Self> {
                Some(Self::constant(f64::MIN))
            }

            fn pi() -> Self {
                Self::constant(std::f64::consts::PI)
            }

            fn two_pi() -> Self {
                Self::constant(2.0 * std::f64::consts::PI)
            }

            fn frac_pi_2() -> Self {
                Self::constant(std::f64::consts::FRAC_PI_2)
            }

            fn frac_pi_3() -> Self {
                Self::constant(std::f64::consts::FRAC_PI_3)
            }

            fn frac_pi_4() -> Self {
                Self::constant(std::f64::consts::FRAC_PI_4)
            }

            fn frac_pi_6() -> Self {
                Self::constant(std::f64::consts::FRAC_PI_6)
            }

            fn frac_pi_8() -> Self {
                Self::constant(std::f64::consts::FRAC_PI_8)
            }

            fn frac_1_pi() -> Self {
                Self::constant(std::f64::consts::FRAC_1_PI)
            }

            fn frac_2_pi() -> Self {
                Self::constant(std::f64::consts::FRAC_2_PI)
            }

            fn frac_2_sqrt_pi() -> Self {
                Self::constant(std::f64::consts::FRAC_2_SQRT_PI)
            }

            fn e() -> Self {
                Self::constant(std::f64::consts::E)
            }

            fn log2_e() -> Self {
                Self::constant(std::f64::consts::LOG2_E)
            }

            fn log10_e() -> Self {
                Self::constant(std::f64::consts::LOG10_E)
            }

            fn ln_2() -> Self {
                Self::constant(std::f64::consts::LN_2)
            }

            fn ln_10() -> Self {
                Self::constant(std::f64::consts::LN_10)
            }
        }

        impl ComplexField for $s {
            type RealField = Self;

            fn from_real(re: Self::RealField) -> Self { re.clone() }

            fn real(self) -> <Self as ComplexField>::RealField { self.clone() }

            fn imaginary(self) -> Self::RealField { Self::zero() }

            fn modulus(self) -> Self::RealField { return ComplexField::abs(self); }

            fn modulus_squared(self) -> Self::RealField { self * self }

            fn argument(self) -> Self::RealField { unimplemented!(); }

            #[inline]
            fn norm1(self) -> Self::RealField { return ComplexField::abs(self); }

            #[inline]
            fn scale(self, factor: Self::RealField) -> Self { return self * factor; }

            #[inline]
            fn unscale(self, factor: Self::RealField) -> Self { return self / factor; }

            #[inline]
            fn floor(self) -> Self {
                Self::new(self.value.floor(), $t::zero())
            }

            #[inline]
            fn ceil(self) -> Self {
                Self::new(self.value.ceil(), $t::zero())
            }

            #[inline]
            fn round(self) -> Self {
                Self::new(self.value.round(), $t::zero())
            }

            #[inline]
            fn trunc(self) -> Self {
                Self::new(self.value.trunc(), $t::zero())
            }

            #[inline]
            fn fract(self) -> Self { Self::new(self.value.fract(), $t::zero()) }

            #[inline]
            fn mul_add(self, a: Self, b: Self) -> Self { return (self * a) + b; }

            #[inline]
            fn abs(self) -> Self::RealField {
                <Self as Signed>::abs(&self)
            }

            #[inline]
            fn hypot(self, other: Self) -> Self::RealField {
                return ComplexField::sqrt(ComplexField::powi(self, 2) + ComplexField::powi(other, 2));
            }

            #[inline]
            fn recip(self) -> Self { return Self::constant(1.0) / self; }

            #[inline]
            fn conjugate(self) -> Self { return self; }

            #[inline]
            fn sin(self) -> Self {
                let output_value = self.value.sin();
                let d_sin_d_arg1 = self.value.cos() as $v;
                let output_tangent = $t::splat(d_sin_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn cos(self) -> Self {
                let output_value = self.value.cos();
                let d_cos_d_arg1 =  (-self.value.sin()) as $v;
                let output_tangent = $t::splat(d_cos_d_arg1)*self.tangent;

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
                let output_value = self.value.tan();
                let c = self.value.cos();
                let d_tan_d_arg1 =  (1.0/(c*c)) as $v;
                let output_tangent = $t::splat(d_tan_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn asin(self) -> Self {
                let output_value = self.value.asin();
                let d_asin_d_arg1 =  (1.0 / (1.0 - self.value * self.value).sqrt()) as $v;
                let output_tangent = $t::splat(d_asin_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn acos(self) -> Self {
                let output_value = self.value.acos();
                let d_acos_d_arg1 =  (-1.0 / (1.0 - self.value * self.value).sqrt()) as $v;
                let output_tangent = $t::splat(d_acos_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn atan(self) -> Self {
                let output_value = self.value.acos();
                let d_atan_d_arg1 =  (1.0 / (self.value * self.value + 1.0)) as $v;
                let output_tangent = $t::splat(d_atan_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn sinh(self) -> Self {
                let output_value = self.value.sinh();
                let d_sinh_d_arg1 =  self.value.cosh() as $v;
                let output_tangent = $t::splat(d_sinh_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn cosh(self) -> Self {
                let output_value = self.value.cosh();
                let d_cosh_d_arg1 =  self.value.sinh() as $v;
                let output_tangent = $t::splat(d_cosh_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn tanh(self) -> Self {
                let output_value = self.value.tanh();
                let c = self.value.cosh();
                let d_tanh_d_arg1 =  (1.0/(c*c)) as $v;
                let output_tangent = $t::splat(d_tanh_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn asinh(self) -> Self {
                let output_value = self.value.asinh();
                let d_asinh_d_arg1 =  (1.0/(self.value*self.value + 1.0).sqrt()) as $v;
                let output_tangent = $t::splat(d_asinh_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn acosh(self) -> Self {
                let output_value = self.value.acosh();
                let d_acosh_d_arg1 =  (1.0/((self.value - 1.0).sqrt()*(self.value + 1.0).sqrt())) as $v;
                let output_tangent = $t::splat(d_acosh_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn atanh(self) -> Self {
                let output_value = self.value.atanh();
                let d_atanh_d_arg1 =  (1.0/(1.0 - self.value*self.value)) as $v;
                let output_tangent = $t::splat(d_atanh_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn log(self, base: Self::RealField) -> Self {
                let output_value = self.value.log(base.value);
                let ln_rhs = base.value.ln();
                let ln_lhs = self.value.ln();
                let d_log_d_arg1 = (1.0/(self.value * ln_rhs)) as $v;
                let d_log_d_arg2 = (-ln_lhs / (base.value * ln_rhs * ln_rhs)) as $v;
                let output_tangent = $t::splat(d_log_d_arg1)*self.tangent + $t::splat(d_log_d_arg2)*base.tangent ;

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
                let output_value = self.value.sqrt();
                let d_sqrt_d_arg1 =  (1.0/(2.0*self.value.sqrt())) as $v;
                let output_tangent = $t::splat(d_sqrt_d_arg1)*self.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn exp(self) -> Self {
                let output_value = self.value.exp();
                let output_tangent = $t::splat(output_value as $v)*self.tangent;

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
                let output_value = self.value.powf(n.value);
                let d_powf_d_arg1 = (n.value * self.value.powf(n.value - 1.0)) as $v;
                let d_powf_d_arg2 = (self.value.powf(n.value) * self.value.ln()) as $v;
                let output_tangent = $t::splat(d_powf_d_arg1)*self.tangent + $t::splat(d_powf_d_arg2)*n.tangent;

                Self {
                    value: output_value,
                    tangent: output_tangent
                }
            }

            #[inline]
            fn powc(self, n: Self) -> Self { return ComplexField::powf(self, n); }

            #[inline]
            fn cbrt(self) -> Self { return ComplexField::powf(self, Self::constant(1.0 / 3.0)); }

            fn is_finite(&self) -> bool { return self.value.is_finite(); }

            fn try_sqrt(self) -> Option<Self> {
                Some(ComplexField::sqrt(self))
            }
        }

        impl SubsetOf<Self> for $s {
            fn to_superset(&self) -> Self {
                self.clone()
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.clone()
            }

            fn is_in_subset(_element: &$s) -> bool {
                true
            }
        }

        impl Field for $s {}

        impl PrimitiveSimdValue for $s {}

        impl SubsetOf<$s> for f32 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as f32
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for f64 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value()
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for u32 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as u32
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for u64 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as u64
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for u128 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as u128
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for i32 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as i32
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for i64 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as i64
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }

        impl SubsetOf<$s> for i128 {
            fn to_superset(&self) -> $s {
                $s::constant(*self as f64)
            }

            fn from_superset_unchecked(element: &$s) -> Self {
                element.value() as i128
            }

            fn is_in_subset(_: &$s) -> bool {
                false
            }
        }
    }
}
make_adf!(f32x2, f32, adf_f32x2, 2);
make_adf!(f32x4, f32, adf_f32x4, 4);
make_adf!(f32x8, f32, adf_f32x8, 8);
make_adf!(f32x16, f32, adf_f32x16, 16);
make_adf!(f64x2, f64, adf_f64x2, 2);
make_adf!(f64x4, f64, adf_f64x4, 4);
make_adf!(f64x8, f64, adf_f64x8, 8);
