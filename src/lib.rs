// #![feature(generic_associated_types)]
// #![feature(const_trait_impl)]

extern crate core;

pub mod differentiable_block;
pub mod forward_ad;
pub mod reverse_ad;
pub mod simd;

use std::cmp::Ordering;
use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use nalgebra::{DefaultAllocator, Dim, DimName, Matrix, OPoint, RawStorageMut};

use num_traits::{Signed};
use simba::scalar::{ComplexField, RealField};
use simba::simd::{SimdComplexField, SimdRealField};
use serde::{Serialize};
use serde::de::DeserializeOwned;

pub trait AD :
    RealField +
    ComplexField +
    PartialOrd +
    PartialEq +
    Signed +
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
    DeserializeOwned
{
    fn constant(constant: f64) -> Self;
    fn to_constant(&self) -> f64;
    fn ad_num_mode() -> ADNumMode;
    fn add_scalar(arg1: f64, arg2: Self) -> Self;
    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn mul_scalar(arg1: f64, arg2: Self) -> Self;
    fn div_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn div_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self;
}


pub trait ObjectAD {
    fn to_constant(&self) -> f64;
}

pub trait NalgebraMatMulAD<'a, R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C> + 'a>:
    AD +
    Mul<Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> +
    Mul<&'a Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> +
    Sized
{ }

pub trait NalgebraPointMulAD<'a, D: DimName>:
    AD +
    Mul<OPoint<Self, D>, Output=OPoint<Self, D>> +
    Mul<&'a OPoint<Self, D>, Output=OPoint<Self, D>> +
    Sized where DefaultAllocator: nalgebra::allocator::Allocator<Self, D>
{ }

pub trait NalgebraMatMulNoRefAD<R: Clone + Dim, C: Clone + Dim, S: Clone + RawStorageMut<Self, R, C>>:
    AD +
    Mul<Matrix<Self, R, C, S>, Output=Matrix<Self, R, C, S>> +
    Sized
{ }

pub trait NalgebraPointMulNoRefAD<D: DimName>:
    AD +
    Mul<OPoint<Self, D>, Output=OPoint<Self, D>> +
    Sized where DefaultAllocator: nalgebra::allocator::Allocator<Self, D>
{ }

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

////////////////////////////////////////////////////////////////////////////////////////////////////

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
}

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
}

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
