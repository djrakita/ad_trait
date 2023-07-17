use std::cmp::Ordering;
use std::fmt;
use std::fmt::{Display, Formatter};
use std::ops::{Add, AddAssign, Div, DivAssign, Index, IndexMut, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign};
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
use nalgebra::{DefaultAllocator, Dim, DimName, Matrix, OPoint, RawStorageMut};
use num_traits::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};
use crate::{AD, ADNumMode, F64};
use serde::{Serialize, Deserialize, Serializer, de, Deserializer};
use serde::de::{MapAccess, Visitor};
use serde::ser::{SerializeStruct};

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

impl<const N: usize> Serialize for f64xn<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer, {
        let mut state = serializer.serialize_struct("adfn", 2)?;
        let value_as_vec: Vec<f64> = self.value.to_vec();
        state.serialize_field("value", &value_as_vec)?;
        state.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for f64xn<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de>, {
        enum Field { Value }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error> where D: Deserializer<'de> {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                        formatter.write_str("value")
                    }

                    fn visit_str<E: de::Error>(self, value: &str) -> Result<Field, E> {
                        match value {
                            "value" => Ok(Field::Value),
                            _ => { Err(de::Error::unknown_field(value, FIELDS)) }
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct F64xnVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for F64xnVisitor<N> {
            type Value = f64xn<N>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("struct f64xn")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<f64xn<N>, V::Error> {
                let mut value = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() { return Err(de::Error::duplicate_field("value")); }
                            let value_as_vec = map.next_value::<Vec<f64>>()?;
                            assert_eq!(value_as_vec.len(), N);
                            let mut value_as_slice = [0.0; N];
                            for (i, t) in value_as_vec.iter().enumerate() { value_as_slice[i] = *t; }
                            value = Some(value_as_slice);
                        }
                    }
                }

                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                Ok(f64xn::<N>{ value })
            }
        }

        const FIELDS: &'static [&'static str] = &["value"];
        deserializer.deserialize_struct("f64xn", FIELDS, F64xnVisitor)
    }
}

impl<const N: usize> AD for f64xn<N> {
    fn constant(constant: f64) -> Self {
        Self::splat(constant)
    }

    fn to_constant(&self) -> f64 {
        panic!("f64xn not compatible with to_constant")
    }

    fn ad_num_mode() -> ADNumMode {
        ADNumMode::SIMDNum
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

impl<const N: usize> Default for f64xn<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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
    fn rem(self, _rhs: Self) -> Self::Output {
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

    fn is_nan(self) -> bool {
        let initial = self.value[0].is_nan();
        if N > 1 {
            for i in 1..N {
                if (self.value[i].is_nan()) != initial {
                    panic!("is_nan is not consistent for f64xn.")
                }
            }
        }
        initial
    }

    fn is_infinite(self) -> bool {
        let initial = self.value[0].is_infinite();
        if N > 1 {
            for i in 1..N {
                if (self.value[i].is_infinite()) != initial {
                    panic!("is_infinite is not consistent for f64xn.")
                }
            }
        }
        initial
    }

    fn is_finite(self) -> bool {
        let initial = self.value[0].is_finite();
        if N > 1 {
            for i in 1..N {
                if (self.value[i].is_finite()) != initial {
                    panic!("is_finite is not consistent for f64xn.")
                }
            }
        }
        initial
    }

    fn is_normal(self) -> bool {
        let initial = self.value[0].is_normal();
        if N > 1 {
            for i in 1..N {
                if (self.value[i].is_normal()) != initial {
                    panic!("is_normal is not consistent for f64xn.")
                }
            }
        }
        initial
    }

    fn classify(self) -> FpCategory {
        let initial = self.value[0].classify();
        if N > 1 {
            for i in 1..N {
                if (self.value[i].classify()) != initial {
                    panic!("classify is not consistent for f64xn.")
                }
            }
        }
        initial
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

    fn integer_decode(self) -> (u64, i16, i8) { panic!("integer_decode not compatible with f64xn") }
}

impl<const N: usize> NumCast for f64xn<N> {
    fn from<T: ToPrimitive>(_n: T) -> Option<Self> { unimplemented!() }
}

impl<const N: usize> ToPrimitive for f64xn<N> {
    fn to_i64(&self) -> Option<i64> {
        None
    }

    fn to_u64(&self) -> Option<u64> {
        None
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize>  PartialEq for f64xn<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        let initial = self.value[0] == other.value[0];
        if N > 1 {
            for i in 1..N {
                if (self.value[i] == other.value[i]) != initial {
                    panic!("eq is not consistent for f64xn.")
                }
            }
        }
        initial
    }
}

impl<const N: usize>  PartialOrd for f64xn<N> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let initial = self.value[0].partial_cmp(&other.value[0]);
        if N > 1 {
            for i in 1..N {
                if (self.value[i].partial_cmp(&other.value[i])) != initial {
                    panic!("partial_cmp is not consistent for f64xn.")
                }
            }
        }
        initial
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
        Self::splat(value)
    }
}
impl<const N: usize>  Into<f64> for f64xn<N> {
    fn into(self) -> f64 {
        self.value[0]
    }
}
impl<const N: usize>  From<f32> for f64xn<N> {
    fn from(value: f32) -> Self {
        Self::splat(value as f64)
    }
}
impl<const N: usize>  Into<f32> for f64xn<N> {
    fn into(self) -> f32 {
        self.value[0] as f32
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize>  UlpsEq for f64xn<N> {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl<const N: usize> AbsDiffEq for f64xn<N> {
    type Epsilon = Self;

    fn default_epsilon() -> Self::Epsilon {
        Self::splat(0.000000001)
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

impl<const N: usize> RelativeEq for f64xn<N> {
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

impl<const N: usize> SimdValue for f64xn<N> {
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

impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64xn<N>, R, C>> Mul<Matrix<f64xn<N>, R, C, S>> for f64xn<N> {
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
impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<Matrix<f64, R, C, S>> for f64xn<N> {
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

impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64xn<N>, R, C>> Mul<&Matrix<f64xn<N>, R, C, S>> for f64xn<N> {
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
impl<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<&Matrix<f64, R, C, S>> for f64xn {
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

impl<const N: usize, D: DimName> Mul<OPoint<f64xn<N>, D>> for f64xn<N> where DefaultAllocator: nalgebra::allocator::Allocator<f64xn<N>, D> {
    type Output = OPoint<f64xn<N>, D>;

    fn mul(self, rhs: OPoint<f64xn<N>, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<const N: usize, D: DimName> Mul<&OPoint<f64xn<N>, D>> for f64xn<N> where DefaultAllocator: nalgebra::allocator::Allocator<f64xn<N>, D> {
    type Output = OPoint<f64xn<N>, D>;

    fn mul(self, rhs: &OPoint<f64xn<N>, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> Zero for f64xn<N> {
    #[inline(always)]
    fn zero() -> Self {
        return Self::constant(0.0)
    }

    fn is_zero(&self) -> bool {
        return *self == Self::constant(0.0);
    }
}

impl<const N: usize> One for f64xn<N> {
    #[inline(always)]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl<const N: usize> Num for f64xn<N> {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = f64::from_str_radix(str, radix).expect("error");
        Ok(Self::constant(val))
    }
}

impl<const N: usize> Signed for f64xn<N> {

    #[inline]
    fn abs(&self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].abs();
        }
        Self {
            value
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
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].signum();
        }
        Self {
            value
        }
    }

    fn is_positive(&self) -> bool {
        return *self > Self::zero();
    }

    fn is_negative(&self) -> bool {
        return *self < Self::zero();
    }
}

impl<const N: usize> FromPrimitive for f64xn<N> {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }
}

impl<const N: usize> Bounded for f64xn<N> {
    fn min_value() -> Self {
        Self::constant(f64::MIN)
    }

    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> RealField for f64xn<N> {
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
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].max(other.value[i]);
        }
        Self { value }
    }

    #[inline]
    fn min(self, other: Self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].min(other.value[i]);
        }
        Self { value }
    }

    #[inline]
    fn clamp(self, min: Self, max: Self) -> Self {
        assert!(min <= max);
        return RealField::min(RealField::max(self, min), max);
    }

    #[inline]
    fn atan2(self, other: Self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].atan2(other.value[i]);
        }
        Self { value }
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

impl<const N: usize> ComplexField for f64xn<N> {
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
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].floor();
        }
        Self { value }
    }

    #[inline]
    fn ceil(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].ceil();
        }
        Self { value }
    }

    #[inline]
    fn round(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].round();
        }
        Self { value }
    }

    #[inline]
    fn trunc(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].trunc();
        }
        Self { value }
    }

    #[inline]
    fn fract(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].fract();
        }
        Self { value }
    }

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
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].sin();
        }
        Self { value }
    }

    #[inline]
    fn cos(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].cos();
        }
        Self { value }
    }

    #[inline]
    fn sin_cos(self) -> (Self, Self) {
        return (ComplexField::sin(self), ComplexField::cos(self));
    }

    #[inline]
    fn tan(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].tan();
        }
        Self { value }
    }

    #[inline]
    fn asin(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].asin();
        }
        Self { value }
    }

    #[inline]
    fn acos(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].acos();
        }
        Self { value }
    }

    #[inline]
    fn atan(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].atan();
        }
        Self { value }
    }

    #[inline]
    fn sinh(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].sinh();
        }
        Self { value }
    }

    #[inline]
    fn cosh(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].cosh();
        }
        Self { value }
    }

    #[inline]
    fn tanh(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].tanh();
        }
        Self { value }
    }

    #[inline]
    fn asinh(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].asinh();
        }
        Self { value }
    }

    #[inline]
    fn acosh(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].acosh();
        }
        Self { value }
    }

    #[inline]
    fn atanh(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].atanh();
        }
        Self { value }
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].log(base.value[i]);
        }
        Self { value }
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
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].sqrt();
        }
        Self { value }
    }

    #[inline]
    fn exp(self) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].exp();
        }
        Self { value }
    }

    #[inline]
    fn exp2(self) -> Self { ComplexField::powf(Self::constant(2.0), self) }

    #[inline]
    fn exp_m1(self) -> Self { return ComplexField::exp(self) - Self::constant(1.0); }

    #[inline]
    fn powi(self, n: i32) -> Self { return ComplexField::powf(self, Self::constant(n as f64)); }

    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        let mut value = [0.0; N];
        for i in 0..N {
            value[i] = self.value[i].powf(n.value[i]);
        }
        Self { value }
    }

    #[inline]
    fn powc(self, n: Self) -> Self { return ComplexField::powf(self, n); }

    #[inline]
    fn cbrt(self) -> Self { return ComplexField::powf(self, Self::constant(1.0 / 3.0)); }

    fn is_finite(&self) -> bool {
        let initial = self.value[0].is_finite();
        if N > 1 {
            for i in 1..N {
                if (self.value[i].is_finite()) != initial {
                    panic!("is_finite is not consistent for f64xn.")
                }
            }
        }
        initial
    }

    fn try_sqrt(self) -> Option<Self> {
        Some(ComplexField::sqrt(self))
    }
}

impl<const N: usize> SubsetOf<Self> for f64xn<N> {
    fn to_superset(&self) -> Self {
        self.clone()
    }

    fn from_superset_unchecked(element: &f64xn<N>) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &f64xn<N>) -> bool {
        true
    }
}

impl<const N: usize> Field for f64xn<N> {}

impl<const N: usize> PrimitiveSimdValue for f64xn<N> {}

impl<const N: usize> SubsetOf<f64xn<N>> for f32 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for f64 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for u32 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for u64 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for u128 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for i32 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for i64 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<f64xn<N>> for i128 {
    fn to_superset(&self) -> f64xn<N> {
        f64xn::constant(*self as f64)
    }

    fn from_superset_unchecked(_element: &f64xn<N>) -> Self {
        panic!("incompatible with f64xn")
    }

    fn is_in_subset(_: &f64xn<N>) -> bool {
        false
    }
}