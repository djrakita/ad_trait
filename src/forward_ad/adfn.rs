use crate::forward_ad::ForwardADTrait;
use crate::{ADNumMode, ADNumType, AD, F64};
use alloc::format;
use alloc::vec::Vec;
use approx::{AbsDiffEq, RelativeEq, UlpsEq};
#[cfg(feature = "bevy")]
use bevy_reflect::Reflect;
use core::cmp::Ordering;
use core::fmt;
use core::fmt::{Display, Formatter};
use core::ops::{
    Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Rem, RemAssign, Sub, SubAssign,
};
use nalgebra::{DefaultAllocator, Dim, DimName, Matrix, OPoint, RawStorageMut};
use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};
use num_traits::{Bounded, FromPrimitive, Num, One, Signed, Zero};
use serde::de::{MapAccess, Visitor};
use serde::ser::SerializeStruct;
use serde::{de, Deserialize, Deserializer, Serialize, Serializer};
use simba::scalar::{ComplexField, Field, RealField, SubsetOf};
use simba::simd::{PrimitiveSimdValue, SimdValue};

/// A type for Forward-mode Automatic Differentiation with multi-tangent support.
///
/// `adfn<N>` stores a value and its `N` associated tangents. This allows for computing
/// up to `N` columns of a Jacobian simultaneously in a single forward pass.
#[allow(non_camel_case_types)]
#[cfg_attr(feature = "bevy", derive(Reflect))]
#[derive(Clone, Debug, Copy)]
pub struct adfn<const N: usize> {
    /// The primary value.
    pub(crate) value: f64,
    /// The tangent values corresponding to the derivatives with respect to input variables.
    pub(crate) tangent: [f64; N],
}
impl<const N: usize> adfn<N> {
    /// Creates a new `adfn` value with its associated tangents.
    pub fn new(value: f64, tangent: [f64; N]) -> Self {
        Self { value, tangent }
    }
    /// Creates a new constant `adfn` value (all tangents are zero).
    pub fn new_constant(value: f64) -> Self {
        Self {
            value,
            tangent: [0.0; N],
        }
    }
    #[inline]
    pub fn value(&self) -> f64 {
        self.value
    }
    #[inline]
    pub fn tangent(&self) -> [f64; N] {
        self.tangent
    }
    #[inline]
    pub fn tangent_size() -> usize {
        N
    }
    #[inline]
    pub fn tangent_as_vec(&self) -> Vec<f64> {
        self.tangent.to_vec()
    }
}

impl<const N: usize> Serialize for adfn<N> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("adfn", 2)?;
        state.serialize_field("value", &self.value)?;
        let tangent_as_vec: Vec<f64> = self.tangent.to_vec();
        state.serialize_field("tangent", &tangent_as_vec)?;
        state.end()
    }
}

impl<'de, const N: usize> Deserialize<'de> for adfn<N> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        enum Field {
            Value,
            Tangent,
        }

        impl<'de> Deserialize<'de> for Field {
            fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
            where
                D: Deserializer<'de>,
            {
                struct FieldVisitor;

                impl<'de> Visitor<'de> for FieldVisitor {
                    type Value = Field;

                    fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                        formatter.write_str("value or tangent")
                    }

                    fn visit_str<E: de::Error>(self, value: &str) -> Result<Field, E> {
                        match value {
                            "value" => Ok(Field::Value),
                            "tangent" => Ok(Field::Tangent),
                            _ => Err(de::Error::unknown_field(value, FIELDS)),
                        }
                    }
                }

                deserializer.deserialize_identifier(FieldVisitor)
            }
        }

        struct AdfnVisitor<const N: usize>;

        impl<'de, const N: usize> Visitor<'de> for AdfnVisitor<N> {
            type Value = adfn<N>;

            fn expecting(&self, formatter: &mut Formatter) -> fmt::Result {
                formatter.write_str("struct adfn")
            }

            fn visit_map<V: MapAccess<'de>>(self, mut map: V) -> Result<adfn<N>, V::Error> {
                let mut value = None;
                let mut tangent = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Value => {
                            if value.is_some() {
                                return Err(de::Error::duplicate_field("value"));
                            }
                            value = Some(map.next_value()?);
                        }
                        Field::Tangent => {
                            if tangent.is_some() {
                                return Err(de::Error::duplicate_field("tangent"));
                            }
                            let tangent_as_vec = map.next_value::<Vec<f64>>()?;
                            assert_eq!(tangent_as_vec.len(), N);
                            let mut tangent_as_slice = [0.0; N];
                            for (i, t) in tangent_as_vec.iter().enumerate() {
                                tangent_as_slice[i] = *t;
                            }
                            tangent = Some(tangent_as_slice);
                        }
                    }
                }

                let value = value.ok_or_else(|| de::Error::missing_field("value"))?;
                let tangent = tangent.ok_or_else(|| de::Error::missing_field("tangent"))?;
                Ok(adfn::<N> { value, tangent })
            }
        }

        const FIELDS: &'static [&'static str] = &["value", "tangent"];
        deserializer.deserialize_struct("adfn", FIELDS, AdfnVisitor)
    }
}

impl<const N: usize> AD for adfn<N> {
    fn constant(constant: f64) -> Self {
        Self {
            value: constant,
            tangent: [0.0; N],
        }
    }

    fn to_constant(&self) -> f64 {
        self.value
    }

    fn ad_num_mode() -> ADNumMode {
        ADNumMode::ForwardAD
    }

    fn ad_num_type() -> ADNumType {
        ADNumType::ADFN
    }

    fn add_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) + arg2
    }

    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self {
        Self::constant(arg1) - arg2
    }

    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 - Self::constant(arg2)
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

    fn mul_by_nalgebra_matrix<
        R: Clone + Dim,
        C: Clone + Dim,
        S: Clone + RawStorageMut<Self, R, C>,
    >(
        &self,
        other: Matrix<Self, R, C, S>,
    ) -> Matrix<Self, R, C, S> {
        *self * other
    }

    fn mul_by_nalgebra_matrix_ref<
        'a,
        R: Clone + Dim,
        C: Clone + Dim,
        S: Clone + RawStorageMut<Self, R, C>,
    >(
        &'a self,
        other: &'a Matrix<Self, R, C, S>,
    ) -> Matrix<Self, R, C, S> {
        *self * other
    }

    fn mul_by_ndarray_matrix_ref<D: Dimension>(
        &self,
        other: &ArrayBase<OwnedRepr<Self>, D>,
    ) -> ArrayBase<OwnedRepr<Self>, D> {
        other * *self
    }
}

impl<const N: usize> ScalarOperand for adfn<N> {}

impl<const N: usize> ForwardADTrait for adfn<N> {
    fn value(&self) -> f64 {
        self.value
    }

    fn tangent_size() -> usize {
        N
    }

    fn tangent_as_vec(&self) -> Vec<f64> {
        self.tangent_as_vec()
    }

    fn set_value(&mut self, value: f64) {
        self.value = value;
    }

    fn set_tangent_value(&mut self, idx: usize, value: f64) {
        self.tangent[idx] = value;
    }
}

impl<const N: usize> Default for adfn<N> {
    fn default() -> Self {
        Self::constant(0.0)
    }
}

/*
impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>> NalgebraMatMulAD2<R, C, S> for adfn<N> {
    fn mul_by_nalgebra_matrix(&self, other: Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        *self * other
    }

    fn mul_by_nalgebra_matrix_ref(&self, other: &Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        *self * other
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

#[inline(always)]
fn two_vecs_mul_and_add<const N: usize>(
    vec1: &[f64; N],
    vec2: &[f64; N],
    scalar1: f64,
    scalar2: f64,
) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = scalar1 * vec1[i] + scalar2 * vec2[i];
    }
    out
}
#[inline(always)]
fn two_vecs_mul_and_add_with_nan_check<const N: usize>(
    vec1: &[f64; N],
    vec2: &[f64; N],
    scalar1: f64,
    scalar2: f64,
) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = mul_with_nan_check(scalar1, vec1[i]) + mul_with_nan_check(scalar2, vec2[i]);
    }
    out
}
#[inline(always)]
fn mul_with_nan_check(a: f64, b: f64) -> f64 {
    return if a.is_nan() && b.is_zero() {
        0.0
    } else if a.is_zero() && b.is_nan() {
        0.0
    } else if a.is_infinite() && b.is_zero() {
        0.0
    } else if a.is_zero() && b.is_infinite() {
        0.0
    } else {
        a * b
    };
}

#[inline(always)]
fn two_vecs_add<const N: usize>(vec1: &[f64; N], vec2: &[f64; N]) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = vec1[i] + vec2[i];
    }
    out
}
#[inline(always)]
fn one_vec_mul<const N: usize>(vec: &[f64; N], scalar: f64) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = scalar * vec[i];
    }
    out
}

#[inline(always)]
fn one_vec_mul_with_nan_check<const N: usize>(vec: &[f64; N], scalar: f64) -> [f64; N] {
    let mut out = [0.0; N];
    for i in 0..N {
        out[i] = mul_with_nan_check(scalar, vec[i]);
    }
    out
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> Add<F64> for adfn<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: F64) -> Self::Output {
        AD::add_scalar(rhs.0, self)
    }
}

impl<const N: usize> AddAssign<F64> for adfn<N> {
    #[inline]
    fn add_assign(&mut self, rhs: F64) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Mul<F64> for adfn<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: F64) -> Self::Output {
        AD::mul_scalar(rhs.0, self)
    }
}

impl<const N: usize> MulAssign<F64> for adfn<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: F64) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Sub<F64> for adfn<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: F64) -> Self::Output {
        AD::sub_r_scalar(self, rhs.0)
    }
}

impl<const N: usize> SubAssign<F64> for adfn<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: F64) {
        *self = *self - rhs;
    }
}

impl<const N: usize> Div<F64> for adfn<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: F64) -> Self::Output {
        AD::div_r_scalar(self, rhs.0)
    }
}

impl<const N: usize> DivAssign<F64> for adfn<N> {
    #[inline]
    fn div_assign(&mut self, rhs: F64) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Rem<F64> for adfn<N> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: F64) -> Self::Output {
        AD::rem_r_scalar(self, rhs.0)
    }
}

impl<const N: usize> RemAssign<F64> for adfn<N> {
    #[inline]
    fn rem_assign(&mut self, rhs: F64) {
        *self = *self % rhs;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> Add<Self> for adfn<N> {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let output_value = self.value + rhs.value;
        let output_tangent = two_vecs_add(&self.tangent, &rhs.tangent);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }
}
impl<const N: usize> AddAssign<Self> for adfn<N> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize> Mul<Self> for adfn<N> {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let output_value = self.value * rhs.value;
        let output_tangent =
            two_vecs_mul_and_add_with_nan_check(&self.tangent, &rhs.tangent, rhs.value, self.value);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }
}
impl<const N: usize> MulAssign<Self> for adfn<N> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize> Sub<Self> for adfn<N> {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        let output_value = self.value - rhs.value;
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &rhs.tangent, 1.0, -1.0);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }
}
impl<const N: usize> SubAssign<Self> for adfn<N> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize> Div<Self> for adfn<N> {
    type Output = Self;

    #[inline]
    fn div(self, rhs: Self) -> Self::Output {
        let output_value = self.value / rhs.value;
        let d_div_d_arg1 = 1.0 / rhs.value;
        let d_div_d_arg2 = -self.value / (rhs.value * rhs.value);
        let output_tangent =
            two_vecs_mul_and_add(&self.tangent, &rhs.tangent, d_div_d_arg1, d_div_d_arg2);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }
}
impl<const N: usize> DivAssign<Self> for adfn<N> {
    #[inline]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl<const N: usize> Rem<Self> for adfn<N> {
    type Output = Self;

    #[inline]
    fn rem(self, rhs: Self) -> Self::Output {
        self - ComplexField::floor(self / rhs) * rhs
        // self - (self / rhs).floor() * rhs
    }
}
impl<const N: usize> RemAssign<Self> for adfn<N> {
    #[inline]
    fn rem_assign(&mut self, rhs: Self) {
        *self = *self % rhs;
    }
}

impl<const N: usize> Neg for adfn<N> {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self {
            value: -self.value,
            tangent: one_vec_mul(&self.tangent, -1.0),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
impl<const N: usize> Float for adf<N> {
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

impl<const N: usize> NumCast for adf<N> {
    fn from<T: ToPrimitive>(_n: T) -> Option<Self> { unimplemented!() }
}

impl<const N: usize> ToPrimitive for adf<N> {
    fn to_i64(&self) -> Option<i64> {
        self.value.to_i64()
    }

    fn to_u64(&self) -> Option<u64> {
        self.value.to_u64()
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> PartialEq for adfn<N> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl<const N: usize> PartialOrd for adfn<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        self.value.partial_cmp(&other.value)
    }
}

impl<const N: usize> Display for adfn<N> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        f.write_str(&format!("{:?}", self)).expect("error");
        Ok(())
    }
}

impl<const N: usize> From<f64> for adfn<N> {
    fn from(value: f64) -> Self {
        Self::new(value, [0.0; N])
    }
}
impl<const N: usize> Into<f64> for adfn<N> {
    fn into(self) -> f64 {
        self.value
    }
}
impl<const N: usize> From<f32> for adfn<N> {
    fn from(value: f32) -> Self {
        Self::new(value as f64, [0.0; N])
    }
}
impl<const N: usize> Into<f32> for adfn<N> {
    fn into(self) -> f32 {
        self.value as f32
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> UlpsEq for adfn<N> {
    fn default_max_ulps() -> u32 {
        unimplemented!("take the time to figure this out.")
    }

    fn ulps_eq(&self, _other: &Self, _epsilon: Self::Epsilon, _max_ulps: u32) -> bool {
        unimplemented!("take the time to figure this out.")
    }
}

impl<const N: usize> AbsDiffEq for adfn<N> {
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

impl<const N: usize> RelativeEq for adfn<N> {
    fn default_max_relative() -> Self::Epsilon {
        Self::constant(0.000000001)
    }

    fn relative_eq(
        &self,
        other: &Self,
        epsilon: Self::Epsilon,
        _max_relative: Self::Epsilon,
    ) -> bool {
        let diff = *self - *other;
        if ComplexField::abs(diff) < epsilon {
            true
        } else {
            false
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> SimdValue for adfn<N> {
    const LANES: usize = 4;
    type Element = Self;
    type SimdBool = bool;

    // fn lanes() -> usize { 4 }

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

impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adfn<N>, R, C>>
    Mul<Matrix<adfn<N>, R, C, S>> for adfn<N>
{
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
impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<f64, R, C>> Mul<Matrix<f64, R, C, S>> for adf<N> {
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

impl<const N: usize, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<adfn<N>, R, C>>
    Mul<&Matrix<adfn<N>, R, C, S>> for adfn<N>
{
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

impl<const N: usize, D: DimName> Mul<OPoint<adfn<N>, D>> for adfn<N>
where
    DefaultAllocator: nalgebra::allocator::Allocator<adfn<N>, D>,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    type Output = OPoint<adfn<N>, D>;

    fn mul(self, rhs: OPoint<adfn<N>, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

impl<const N: usize, D: DimName> Mul<&OPoint<adfn<N>, D>> for adfn<N>
where
    DefaultAllocator: nalgebra::allocator::Allocator<adfn<N>, D>,
    DefaultAllocator: nalgebra::allocator::Allocator<D>,
{
    type Output = OPoint<adfn<N>, D>;

    fn mul(self, rhs: &OPoint<adfn<N>, D>) -> Self::Output {
        let mut out_clone = rhs.clone();
        for e in out_clone.iter_mut() {
            *e *= self;
        }
        out_clone
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> Zero for adfn<N> {
    #[inline(always)]
    fn zero() -> Self {
        return Self::constant(0.0);
    }

    fn is_zero(&self) -> bool {
        return self.value == 0.0;
    }
}

impl<const N: usize> One for adfn<N> {
    #[inline(always)]
    fn one() -> Self {
        Self::constant(1.0)
    }
}

impl<const N: usize> Num for adfn<N> {
    type FromStrRadixErr = ();

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let val = f64::from_str_radix(str, radix).expect("error");
        Ok(Self::constant(val))
    }
}

impl<const N: usize> Signed for adfn<N> {
    #[inline]
    fn abs(&self) -> Self {
        let output_value = self.value.abs();
        let output_tangent = if self.value >= 0.0 {
            self.tangent
        } else {
            one_vec_mul(&self.tangent, -1.0)
        };

        Self {
            value: output_value,
            tangent: output_tangent,
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
        let output_tangent = [0.0; N];
        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    fn is_positive(&self) -> bool {
        return self.value > 0.0;
    }

    fn is_negative(&self) -> bool {
        return self.value < 0.0;
    }
}

impl<const N: usize> FromPrimitive for adfn<N> {
    fn from_i64(n: i64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }

    fn from_u64(n: u64) -> Option<Self> {
        Some(Self::constant(n as f64))
    }
}

impl<const N: usize> Bounded for adfn<N> {
    fn min_value() -> Self {
        Self::constant(f64::MIN)
    }

    fn max_value() -> Self {
        Self::constant(f64::MAX)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<const N: usize> RealField for adfn<N> {
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
            tangent: output_tangent,
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
            tangent: output_tangent,
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
        let d_atan2_d_arg1 = other.value / (self.value * self.value + other.value * other.value);
        let d_atan2_d_arg2 = -self.value / (self.value * self.value + other.value * other.value);
        let output_tangent = two_vecs_mul_and_add(
            &self.tangent,
            &other.tangent,
            d_atan2_d_arg1,
            d_atan2_d_arg2,
        );

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    fn min_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }

    fn max_value() -> Option<Self> {
        Some(Self::constant(f64::MIN))
    }

    fn pi() -> Self {
        Self::constant(core::f64::consts::PI)
    }

    fn two_pi() -> Self {
        Self::constant(2.0 * core::f64::consts::PI)
    }

    fn frac_pi_2() -> Self {
        Self::constant(core::f64::consts::FRAC_PI_2)
    }

    fn frac_pi_3() -> Self {
        Self::constant(core::f64::consts::FRAC_PI_3)
    }

    fn frac_pi_4() -> Self {
        Self::constant(core::f64::consts::FRAC_PI_4)
    }

    fn frac_pi_6() -> Self {
        Self::constant(core::f64::consts::FRAC_PI_6)
    }

    fn frac_pi_8() -> Self {
        Self::constant(core::f64::consts::FRAC_PI_8)
    }

    fn frac_1_pi() -> Self {
        Self::constant(core::f64::consts::FRAC_1_PI)
    }

    fn frac_2_pi() -> Self {
        Self::constant(core::f64::consts::FRAC_2_PI)
    }

    fn frac_2_sqrt_pi() -> Self {
        Self::constant(core::f64::consts::FRAC_2_SQRT_PI)
    }

    fn e() -> Self {
        Self::constant(core::f64::consts::E)
    }

    fn log2_e() -> Self {
        Self::constant(core::f64::consts::LOG2_E)
    }

    fn log10_e() -> Self {
        Self::constant(core::f64::consts::LOG10_E)
    }

    fn ln_2() -> Self {
        Self::constant(core::f64::consts::LN_2)
    }

    fn ln_10() -> Self {
        Self::constant(core::f64::consts::LN_10)
    }
}

impl<const N: usize> ComplexField for adfn<N> {
    type RealField = Self;

    fn from_real(re: Self::RealField) -> Self {
        re.clone()
    }

    fn real(self) -> <Self as ComplexField>::RealField {
        self.clone()
    }

    fn imaginary(self) -> Self::RealField {
        Self::zero()
    }

    fn modulus(self) -> Self::RealField {
        return ComplexField::abs(self);
    }

    fn modulus_squared(self) -> Self::RealField {
        self * self
    }

    fn argument(self) -> Self::RealField {
        unimplemented!();
    }

    #[inline]
    fn norm1(self) -> Self::RealField {
        return ComplexField::abs(self);
    }

    #[inline]
    fn scale(self, factor: Self::RealField) -> Self {
        return self * factor;
    }

    #[inline]
    fn unscale(self, factor: Self::RealField) -> Self {
        return self / factor;
    }

    #[inline]
    fn floor(self) -> Self {
        Self::new(self.value.floor(), [0.0; N])
    }

    #[inline]
    fn ceil(self) -> Self {
        Self::new(self.value.ceil(), [0.0; N])
    }

    #[inline]
    fn round(self) -> Self {
        Self::new(self.value.round(), [0.0; N])
    }

    #[inline]
    fn trunc(self) -> Self {
        Self::new(self.value.trunc(), [0.0; N])
    }

    #[inline]
    fn fract(self) -> Self {
        Self::new(self.value.fract(), [1.0; N])
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
        // return ComplexField::sqrt(ComplexField::powi(self, 2) + ComplexField::powi(other, 2));
        return ComplexField::sqrt(self * self + other * other);
    }

    #[inline]
    fn recip(self) -> Self {
        return Self::constant(1.0) / self;
    }

    #[inline]
    fn conjugate(self) -> Self {
        return self;
    }

    #[inline]
    fn sin(self) -> Self {
        let output_value = self.value.sin();
        let d_sin_d_arg1 = self.value.cos();
        let output_tangent = one_vec_mul(&self.tangent, d_sin_d_arg1);
        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn cos(self) -> Self {
        let output_value = self.value.cos();
        let d_cos_d_arg1 = -self.value.sin();
        let output_tangent = one_vec_mul(&self.tangent, d_cos_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
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
        let d_tan_d_arg1 = 1.0 / (c * c);
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_tan_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn asin(self) -> Self {
        let output_value = self.value.asin();
        let d_asin_d_arg1 = 1.0 / (1.0 - self.value * self.value).sqrt();
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_asin_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn acos(self) -> Self {
        let output_value = self.value.acos();
        let d_acos_d_arg1 = -1.0 / (1.0 - self.value * self.value).sqrt();
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_acos_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn atan(self) -> Self {
        let output_value = self.value.atan();
        let d_atan_d_arg1 = 1.0 / (self.value * self.value + 1.0);
        let output_tangent = one_vec_mul(&self.tangent, d_atan_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn sinh(self) -> Self {
        let output_value = self.value.sinh();
        let d_sinh_d_arg1 = self.value.cosh();
        let output_tangent = one_vec_mul(&self.tangent, d_sinh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn cosh(self) -> Self {
        let output_value = self.value.cosh();
        let d_cosh_d_arg1 = self.value.sinh();
        let output_tangent = one_vec_mul(&self.tangent, d_cosh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn tanh(self) -> Self {
        let output_value = self.value.tanh();
        let c = self.value.cosh();
        let d_tanh_d_arg1 = 1.0 / (c * c);
        let output_tangent = one_vec_mul(&self.tangent, d_tanh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn asinh(self) -> Self {
        let output_value = self.value.asinh();
        let d_asinh_d_arg1 = 1.0 / (self.value * self.value + 1.0).sqrt();
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_asinh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn acosh(self) -> Self {
        let output_value = self.value.acosh();
        let d_acosh_d_arg1 = 1.0 / (self.value * self.value - 1.0).sqrt();
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_acosh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn atanh(self) -> Self {
        let output_value = self.value.atanh();
        let d_atanh_d_arg1 = 1.0 / (1.0 - self.value * self.value);
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_atanh_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn log(self, base: Self::RealField) -> Self {
        let output_value = self.value.log(base.value);
        let ln_rhs = base.value.ln();
        let ln_lhs = self.value.ln();
        let d_log_d_arg1 = 1.0 / (self.value * ln_rhs);
        let d_log_d_arg2 = -ln_lhs / (base.value * ln_rhs * ln_rhs);
        let output_tangent = two_vecs_mul_and_add_with_nan_check(
            &self.tangent,
            &base.tangent,
            d_log_d_arg1,
            d_log_d_arg2,
        );

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn log2(self) -> Self {
        return ComplexField::log(self, Self::constant(2.0));
    }

    #[inline]
    fn log10(self) -> Self {
        return ComplexField::log(self, Self::constant(10.0));
    }

    #[inline]
    fn ln(self) -> Self {
        return ComplexField::log(self, Self::constant(core::f64::consts::E));
    }

    #[inline]
    fn ln_1p(self) -> Self {
        ComplexField::ln(Self::constant(1.0) + self)
    }

    #[inline]
    fn sqrt(self) -> Self {
        let output_value = self.value.sqrt();
        let tmp = if self.value == 0.0 {
            0.0001
        } else {
            self.value
        };
        let d_sqrt_d_arg1 = 1.0 / (2.0 * tmp.sqrt());
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, d_sqrt_d_arg1);

        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn exp(self) -> Self {
        let output_value = self.value.exp();
        let output_tangent = one_vec_mul_with_nan_check(&self.tangent, output_value);
        Self {
            value: output_value,
            tangent: output_tangent,
        }
    }

    #[inline]
    fn exp2(self) -> Self {
        ComplexField::powf(Self::constant(2.0), self)
    }

    #[inline]
    fn exp_m1(self) -> Self {
        return ComplexField::exp(self) - Self::constant(1.0);
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        return ComplexField::powf(self, Self::constant(n as f64));
    }

    #[inline]
    fn powf(self, n: Self::RealField) -> Self {
        let output_value = self.value.powf(n.value) as f64;
        let d_powf_d_arg1 = n.value * self.value.powf(n.value - 1.0) as f64;
        let tmp = if self.value == 0.0 {
            0.0001
        } else {
            self.value
        };
        let d_powf_d_arg2 = (output_value * tmp.ln()) as f64;
        let output_tangent = two_vecs_mul_and_add_with_nan_check(
            &self.tangent,
            &n.tangent,
            d_powf_d_arg1,
            d_powf_d_arg2,
        );
        return Self {
            value: output_value,
            tangent: output_tangent,
        };
        /*
        let output_value = self.value.powf(n.value);
        let d_powf_d_arg1 = n.value * self.value.powf(n.value - 1.0);
        let d_powf_d_arg2 = if self.value < 0.0 {
            0.0
        } else {
            self.value.powf(n.value) * self.value.ln()
        };
        let output_tangent = two_vecs_mul_and_add(&self.tangent, &n.tangent, d_powf_d_arg1, d_powf_d_arg2);

        Self {
            value: output_value,
            tangent: output_tangent
        }
        */
    }

    #[inline]
    fn powc(self, n: Self) -> Self {
        return ComplexField::powf(self, n);
    }

    #[inline]
    fn cbrt(self) -> Self {
        return ComplexField::powf(self, Self::constant(1.0 / 3.0));
    }

    fn is_finite(&self) -> bool {
        return self.value.is_finite();
    }

    fn try_sqrt(self) -> Option<Self> {
        Some(ComplexField::sqrt(self))
    }
}

impl<const N: usize> SubsetOf<Self> for adfn<N> {
    fn to_superset(&self) -> Self {
        self.clone()
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.clone()
    }

    fn is_in_subset(_element: &adfn<N>) -> bool {
        true
    }
}

impl<const N: usize> Field for adfn<N> {}

impl<const N: usize> PrimitiveSimdValue for adfn<N> {}

impl<const N: usize> SubsetOf<adfn<N>> for f32 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as f32
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for f64 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value()
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for u32 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as u32
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for u64 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as u64
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for u128 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as u128
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for i32 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as i32
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for i64 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as i64
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

impl<const N: usize> SubsetOf<adfn<N>> for i128 {
    fn to_superset(&self) -> adfn<N> {
        adfn::new_constant(*self as f64)
    }

    fn from_superset_unchecked(element: &adfn<N>) -> Self {
        element.value() as i128
    }

    fn is_in_subset(_: &adfn<N>) -> bool {
        false
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

unsafe impl<const N: usize> Dim for adfn<N> {
    fn try_to_usize() -> Option<usize> {
        unimplemented!()
    }

    fn value(&self) -> usize {
        unimplemented!()
    }

    fn from_usize(_dim: usize) -> Self {
        unimplemented!()
    }
}
