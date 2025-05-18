
//! # Introduction
//! This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
//! Rust programming language. Utilizing Rust's extensive and expressive trait features, the several
//! types in this crate that implement the trait AD can be thought of as a drop-in replacement for an
//! f64 or f32 that affords forward mode or backwards mode automatic differentiation on any downstream
//! computation in Rust.
//!
//! # Key Features
//! - ad_trait supports reverse mode or forward mode automatic differentiation. The forward mode automatic
//! differentiation implementation can also take advantage of SIMD to compute multiple tangents simultaneously.
//! - The core rust f64 or f32 types also implement the AD trait, meaning any functions that take an AD
//! trait object as a generic type can handle either standard floating point computation or derivative
//! tracking automatic differentiation with essentially no overhead.
//! - The provided types that implement the AD trait also implement several useful traits that allow it
//! to operate almost exactly as a standard f64. For example, it even implements the `RealField` and
//! `ComplexField` traits, meaning it can be used in any `nalgebra` or `ndarray` computations.
//!
//! # Example
//! ```
//! use ad_trait::AD;
//! use ad_trait::function_engine::FunctionEngine;
//! use ad_trait::differentiable_function::{DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, ReverseAD};
//! use ad_trait::forward_ad::adfn::adfn;
//! use ad_trait::reverse_ad::adr::adr;
//!
//! #[derive(Clone)]
//! pub struct Test<T: AD> {
//!     coeff: T
//! }
//! impl<T: AD> DifferentiableFunctionTrait<T> for Test<T> {
//!     const NAME: &'static str = "Test";
//!
//!     fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
//!         vec![ self.coeff*inputs[0].sin() + inputs[1].cos() ]
//!     }
//!
//!     fn num_inputs(&self) -> usize {
//!         2
//!     }
//!
//!     fn num_outputs(&self) -> usize {
//!         1
//!     }
//! }
//! impl<T: AD> Test<T> {
//!     pub fn to_other_ad_type<T2: AD>(&self) -> Test<T2> {
//!         Test { coeff: self.coeff.to_other_ad_type::<T2>() }
//!     }
//! }
//!
//!
//! fn main() {
//!     let inputs = vec![1., 2.];
//!
//!     // Reverse AD //////////////////////////////////////////////////////////////////////////////////
//!     let function_standard = Test { coeff: 2.0 };
//!     let function_derivative = function_standard.to_other_ad_type::<adr>();
//!     let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ReverseAD::new());
//!
//!     let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
//!     println!("Reverse AD: ");
//!     println!("  f_res: {}", f_res[0]);
//!     println!("  derivative: {}", derivative_res);
//!     println!("//////////////");
//!     println!();
//!
//!     // Forward AD //////////////////////////////////////////////////////////////////////////////////
//!     let function_standard = Test { coeff: 2.0 };
//!     let function_derivative = function_standard.to_other_ad_type::<adfn<1>>();
//!     let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardAD::new());
//!
//!     let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
//!     println!("Forward AD: ");
//!     println!("  f_res: {}", f_res[0]);
//!     println!("  derivative: {}", derivative_res);
//!     println!("//////////////");
//!     println!();
//!
//!     // Forward AD Multi ////////////////////////////////////////////////////////////////////////////
//!     let function_standard = Test { coeff: 2.0 };
//!     let function_derivative = function_standard.to_other_ad_type::<adfn<2>>();
//!     let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardADMulti::new());
//!
//!     let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
//!     println!("Forward AD Multi: ");
//!     println!("  f_res: {}", f_res[0]);
//!     println!("  derivative: {}", derivative_res);
//!     println!("//////////////");
//!     println!();
//!
//!     // Finite Differencing /////////////////////////////////////////////////////////////////////////
//!     let function_standard = Test { coeff: 2.0 };
//!     let function_derivative = function_standard.clone();
//!     let differentiable_block = FunctionEngine::new(function_standard, function_derivative, FiniteDifferencing::new());
//!
//!     let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
//!     println!("Finite Differencing: ");
//!     println!("  f_res: {}", f_res[0]);
//!     println!("  derivative: {}", derivative_res);
//!     println!("//////////////");
//!     println!();
//!
//! }
//! ```
//! # Citation
//!
//! For more information about our work, refer to our paper:
//! https://arxiv.org/abs/2504.15976
//!
//! If you use this crate in your research, please cite:
//!
//! ```text
//! @article{liang2025ad,
//!   title={ad-trait: A Fast and Flexible Automatic Differentiation Library in Rust},
//!   author={Liang, Chen and Wang, Qian and Xu, Andy and Rakita, Daniel},
//!   journal={arXiv preprint arXiv:2504.15976},
//!   year={2025}
//! }
//! ```
//!

// #![feature(min_specialization)]
// #![feature(portable_simd)]
// #![feature(trivial_bounds)]
#![cfg_attr(feature = "nightly", feature(trivial_bounds, portable_simd, min_specialization))]
extern crate core;

pub mod differentiable_function;
pub mod function_engine;
pub mod forward_ad;
pub mod reverse_ad;
pub mod simd;

use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use bevy_reflect::Reflect;
use nalgebra::{Dim, Matrix, RawStorageMut, Scalar};
use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};
use num_traits::Signed;
use simba::scalar::{ComplexField, RealField};
use simba::simd::{SimdComplexField, SimdRealField};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde::de::DeserializeOwned;
use serde_with::{DeserializeAs, SerializeAs};

pub trait AD :
    RealField +
    ComplexField +
    PartialOrd +
    PartialEq +
    Signed +
    Scalar +
    // Float +
    Clone +
    Copy +
    Debug +
    Display +
    Default +

    Add<F64, Output=Self> +
    AddAssign<F64> +
    Mul<F64, Output=Self> +
    MulAssign<F64> +
    Sub<F64, Output=Self> +
    SubAssign<F64> +
    Div<F64, Output=Self> +
    DivAssign<F64> +
    Rem<F64, Output=Self> +
    RemAssign<F64> +

    From<f32> +
    Into<f64> +

    SimdRealField +
    SimdComplexField +

    Serialize +
    DeserializeOwned +

    Reflect +

    ScalarOperand
{
    fn constant(constant: f64) -> Self;
    fn to_constant(&self) -> f64;
    #[inline(always)]
    fn to_constant_ad(&self) -> Self {
        Self::constant(self.to_constant())
    }
    fn ad_num_mode() -> ADNumMode;
    fn ad_num_type() -> ADNumType;
    fn add_scalar(arg1: f64, arg2: Self) -> Self;
    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn mul_scalar(arg1: f64, arg2: Self) -> Self;
    fn div_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn div_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn mul_by_nalgebra_matrix<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>(&self, other: Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S>;
    fn mul_by_nalgebra_matrix_ref<'a, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>(&'a self, other: &'a Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S>;
    fn mul_by_ndarray_matrix_ref<D: Dimension>(&self, other: &ArrayBase<OwnedRepr<Self>, D>) -> ArrayBase<OwnedRepr<Self>, D>;
    fn to_other_ad_type<T2: AD>(&self) -> T2 {
        T2::constant(self.to_constant())
    }
}

pub trait ObjectAD {
    fn to_constant(&self) -> f64;
}

/*
pub trait NalgebraMatMulAD<'a, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C> + 'a>:
    AD +
    Mul<Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> +
    Mul<&'a Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> +
    Sized
{ }
*/

/*
pub trait NalgebraMatMulAD<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>:
    Sized
{
    fn mul_by_nalgebra_matrix(&self, other: Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S>;
    fn mul_by_nalgebra_matrix_ref<'a>(&'a self, other: &'a Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S>;
}

impl<T: AD, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>> NalgebraMatMulAD<R, C, S> for T
    where T: Mul<Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> + for<'a> Mul<&'a Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>>
{
    fn mul_by_nalgebra_matrix(&self, other: Matrix<T, R, C, S>) -> Matrix<T, R, C, S> {
        *self * other
    }
    fn mul_by_nalgebra_matrix_ref<'b>(&'b self, other: &'b Matrix<T, R, C, S>) -> Matrix<T, R, C, S> {
        *self * other
    }
}
*/

/*
pub trait NalgebraPointMulAD<'a, D: DimName>:
    AD +
    Mul<OPoint<Self, D>, Output=OPoint<Self, D>> +
    Mul<&'a OPoint<Self, D>, Output=OPoint<Self, D>> +
    Sized where DefaultAllocator: nalgebra::allocator::Allocator<Self, D>
{ }
*/
// pub type OVector<T, D> = Matrix<T, D, U1, Owned<T, D, U1>>;
// <DefaultAllocator as Allocator<T, R, C>>::Buffer

/*
pub trait NalgebraMatMulAD3<R: Clone + Dim>:
    AD +
    Mul<Matrix<Self, R, U1, Owned<Self, R, U1>>, Output=Matrix<Self, R, U1, Owned<Self, R, U1>>> +
    Sized
    where DefaultAllocator: Allocator<Self, R>
{
    fn mul(&self, other: Matrix<Self, R, U1, Owned<Self, R, U1>>) -> Matrix<Self, R, U1, Owned<Self, R, U1>>;
    fn mul_by_ref(&self, other: &Matrix<Self, R, U1, Owned<Self, R, U1>>) -> Matrix<Self, R, U1, Owned<Self, R, U1>>;
}
*/

/*
pub trait NalgebraPointMulAD2<D: DimName>:
    AD +
    // Mul<OPoint<Self, D>, Output=OPoint<Self, D>> +
    Sized
    where DefaultAllocator: Allocator<Self, D>
{
    fn mul(&self, other: OPoint<Self, D>) -> OPoint<Self, D>;
    fn mul_by_ref(&self, other: &OPoint<Self, D>) -> OPoint<Self, D>;
}
*/

/*
pub trait NalgebraMatMulNoRefAD<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>:
    AD +
    Mul<Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> +
    Sized
{ }
*/

/*
pub trait NalgebraPointMulNoRefAD<D: DimName>:
    AD +
    Mul<OPoint<Self, D>, Output=OPoint<Self, D>> +
    Sized where DefaultAllocator: nalgebra::allocator::Allocator<Self, D>
{ }
*/

#[macro_export]
macro_rules! ad_setup {
    ($($T: ident),*) => {
        $(
        ad_setup_f64!($T);
        // ad_setup_any_nalgebra_dmatrix!($T);
        )*
    }
}
ad_setup!(f64, f32);

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ADNumMode {
    Float,
    ForwardAD,
    ReverseAD,
    SIMDNum
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ADNumType {
    F64,
    F32,
    ADR,
    ADFN,
    ADF,
    F64XN
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub trait FloatADTrait: AD { }

impl AD for f64 {
    fn constant(v: f64) -> Self {
        return v;
    }

    fn to_constant(&self) -> f64 {
        *self
    }

    fn ad_num_mode() -> ADNumMode {
        ADNumMode::Float
    }

    fn ad_num_type() -> ADNumType {
        ADNumType::F64
    }

    fn add_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 + arg2
    }

    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 - arg2
    }

    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 - arg2
    }

    fn mul_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 * arg2
    }

    fn div_l_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 / arg2
    }

    fn div_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 / arg2
    }

    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 % arg2
    }

    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 % arg2
    }

    fn mul_by_nalgebra_matrix<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>(&self, other: Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = other.clone();
        out.iter_mut().for_each(|x| *x *= *self);
        out
    }

    fn mul_by_nalgebra_matrix_ref<'a, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>(&'a self, other: &'a Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = other.clone();
        out.iter_mut().for_each(|x| *x *= *self);
        out
    }

    fn mul_by_ndarray_matrix_ref<D: Dimension>(&self, other: &ArrayBase<OwnedRepr<Self>, D>) -> ArrayBase<OwnedRepr<Self>, D> {
        other * *self
    }
}
impl FloatADTrait for f64 { }

impl AD for f32 {
    fn constant(v: f64) -> Self {
        return v as f32;
    }

    fn to_constant(&self) -> f64 {
        *self as f64
    }

    fn ad_num_mode() -> ADNumMode {
        ADNumMode::Float
    }

    fn ad_num_type() -> ADNumType {
        ADNumType::F32
    }

    fn add_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 + arg2
    }

    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 - arg2
    }

    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 - arg2 as f32
    }

    fn mul_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 * arg2
    }

    fn div_l_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 / arg2
    }

    fn div_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 / arg2 as f32
    }

    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 % arg2
    }

    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self {
        arg1 % arg2 as f32
    }

    fn mul_by_nalgebra_matrix<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>(&self, other: Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = other.clone();
        out.iter_mut().for_each(|x| *x *= *self);
        out
    }

    fn mul_by_nalgebra_matrix_ref<'a, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>(&'a self, other: &'a Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = other.clone();
        out.iter_mut().for_each(|x| *x *= *self);
        out
    }

    fn mul_by_ndarray_matrix_ref<D: Dimension>(&self, other: &ArrayBase<OwnedRepr<Self>, D>) -> ArrayBase<OwnedRepr<Self>, D> {
        other * *self
    }
}
impl FloatADTrait for f32 { }

/*
#[macro_export]
macro_rules! nalgebra_mat_mul_ad_setup {
    ($t1: tt, $t2: tt; $(($x:tt, $y:tt)),*) => {
        $(
            impl NalgebraMatMulAD2<Const<$x>, Const<$y>, ArrayStorage<$t1, $x, $y>> for $t1 {
                fn mul_by_nalgebra_matrix(&self, other: Matrix<$t1, Const<$x>, Const<$y>, ArrayStorage<$t1, $x, $y>>) -> Matrix<$t1, Const<$x>, Const<$y>, ArrayStorage<$t1, $x, $y>> {
                    *self * other
                }
                fn mul_by_nalgebra_matrix_ref(&self, other: &Matrix<$t1, Const<$x>, Const<$y>, ArrayStorage<$t1, $x, $y>>) -> Matrix<$t1, Const<$x>, Const<$y>, ArrayStorage<$t1, $x, $y>> {
                    *self * other
                }
            }
            impl NalgebraMatMulAD2<Const<$x>, Const<$y>, ArrayStorage<$t2, $x, $y>> for $t2 {
                fn mul_by_nalgebra_matrix(&self, other: Matrix<$t2, Const<$x>, Const<$y>, ArrayStorage<$t2, $x, $y>>) -> Matrix<$t2, Const<$x>, Const<$y>, ArrayStorage<$t2, $x, $y>> {
                    *self * other
                }
                fn mul_by_nalgebra_matrix_ref(&self, other: &Matrix<$t2, Const<$x>, Const<$y>, ArrayStorage<$t2, $x, $y>>) -> Matrix<$t2, Const<$x>, Const<$y>, ArrayStorage<$t2, $x, $y>> {
                    *self * other
                }
            }
        )*
    }
}
nalgebra_mat_mul_ad_setup!(f64, f32; (1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2), (1, 1), (2, 2), (3, 3), (4, 4));
*/

// impl<'a> NalgebraMatMulAD<'a, Const<3>, Const<1>, ArrayStorage<f64, 3, 1>> for f64 { }
// impl<'a> NalgebraMatMulAD<'a, Const<3>, Const<3>, ArrayStorage<f64, 3, 3>> for f64 { }

// impl<'a> NalgebraMatMulAD<'a, Const<3>, Const<1>, ArrayStorage<f32, 3, 1>> for f32 { }
// impl<'a> NalgebraMatMulAD<'a, Const<3>, Const<1>, ArrayStorage<f32, 3, 1>> for f32 { }

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Copy)]
pub struct F64(pub f64);

impl<T: AD> Add<T> for F64 {
    type Output = T;

    #[inline]
    fn add(self, rhs: T) -> Self::Output {
        AD::add_scalar(self.0, rhs)
    }
}
impl<T: AD> Mul<T> for F64 {
    type Output = T;

    fn mul(self, rhs: T) -> Self::Output {
        AD::mul_scalar(self.0, rhs)
    }
}
impl<T: AD> Sub<T> for F64 {
    type Output = T;

    fn sub(self, rhs: T) -> Self::Output {
        AD::sub_l_scalar(self.0, rhs)
    }
}
impl<T: AD> Div<T> for F64 {
    type Output = T;

    fn div(self, rhs: T) -> Self::Output {
        AD::div_l_scalar(self.0, rhs)
    }
}
impl<T: AD> Rem<T> for F64 {
    type Output = T;

    fn rem(self, rhs: T) -> Self::Output {
        AD::rem_l_scalar(self.0, rhs)
    }
}

#[macro_export]
macro_rules! ad_setup_f64 {
    ($T: ident) => {
        impl Add<F64> for $T {
            type Output = $T;

            #[inline]
            fn add(self, rhs: F64) -> Self::Output {
                AD::add_scalar(rhs.0, self)
            }
        }

        impl AddAssign<F64> for $T {
            #[inline]
            fn add_assign(&mut self, rhs: F64) {
                *self = *self + rhs;
            }
        }

        impl Mul<F64> for $T {
            type Output = $T;

            #[inline]
            fn mul(self, rhs: F64) -> Self::Output {
                AD::mul_scalar(rhs.0, self)
            }
        }

        impl MulAssign<F64> for $T {
            #[inline]
            fn mul_assign(&mut self, rhs: F64) {
                *self = *self * rhs;
            }
        }

        impl Sub<F64> for $T {
            type Output = $T;

            #[inline]
            fn sub(self, rhs: F64) -> Self::Output {
                AD::sub_r_scalar(self, rhs.0)
            }
        }

        impl SubAssign<F64> for $T {
            #[inline]
            fn sub_assign(&mut self, rhs: F64) {
                *self = *self - rhs;
            }
        }

        impl Div<F64> for $T {
            type Output = $T;

            #[inline]
            fn div(self, rhs: F64) -> Self::Output {
                AD::div_r_scalar(self, rhs.0)
            }
        }

        impl DivAssign<F64> for $T {
            #[inline]
            fn div_assign(&mut self, rhs: F64) {
                *self = *self / rhs;
            }
        }

        impl Rem<F64> for $T {
            type Output = $T;

            #[inline]
            fn rem(self, rhs: F64) -> Self::Output {
                AD::rem_r_scalar(self, rhs.0)
            }
        }

        impl RemAssign<F64> for $T {
            #[inline]
            fn rem_assign(&mut self, rhs: F64) {
                *self = *self % rhs;
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<T: AD> ObjectAD for T {
    fn to_constant(&self) -> f64 {
        self.to_constant()
    }
}

impl PartialEq<f64> for dyn ObjectAD {
    fn eq(&self, other: &f64) -> bool {
        self.to_constant().eq(other)
    }
}

impl PartialOrd<f64> for dyn ObjectAD {
    fn partial_cmp(&self, other: &f64) -> Option<Ordering> {
        self.to_constant().partial_cmp(other)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Custom serializer for the AD trait
pub fn ad_custom_serialize<S, T: AD>(value: &T, serializer: S) -> Result<S::Ok, S::Error>
where
    S: Serializer,
{
    serializer.serialize_f64(value.to_constant())
}

// Custom deserializer for the AD trait
pub fn ad_custom_deserialize<'de, D, T: AD>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
{
    let constant = f64::deserialize(deserializer)?;
    Ok(T::constant(constant))
}
pub struct SerdeAD<T: AD>(pub T);

impl<T: AD> SerializeAs<T> for SerdeAD<T> {
    fn serialize_as<S>(source: &T, serializer: S) -> Result<S::Ok, S::Error> where S: Serializer {
        ad_custom_serialize(source, serializer)
    }
}
impl<'de, T: AD> DeserializeAs<'de, T> for SerdeAD<T> {
    fn deserialize_as<D>(deserializer: D) -> Result<T, D::Error> where D: Deserializer<'de> {
        ad_custom_deserialize(deserializer)
    }
}

pub trait ADConvertableTrait {
    type ConvertableType<T: AD>;

    fn convert_to_other_ad_type<T1: AD, T2: AD>(input: &Self::ConvertableType<T1>) -> Self::ConvertableType<T2>;
}
impl ADConvertableTrait for () {
    type ConvertableType<T: AD> = ();

    fn convert_to_other_ad_type<T1: AD, T2: AD>(_input: &Self::ConvertableType<T1>) -> Self::ConvertableType<T2> { () }
}
