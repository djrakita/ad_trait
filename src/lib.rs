extern crate core;

pub mod differentiable_block;
pub mod forward_ad;
pub mod reverse_ad;
pub mod simd;

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Rem, RemAssign, Sub, SubAssign};
use faer_core::Entity;
use nalgebra::{DMatrix, DVector};
use num_traits::{Float, Signed};
use simba::scalar::ComplexField;


pub trait AD :
    ComplexField +
    PartialOrd +
    PartialEq +
    Signed +
    // Float +
    Clone +
    Copy +
    Debug +
    Display +

    Add<F64, Output=Self> +
    AddAssign<F64> +
    Mul<F64, Output=Self> +
    MulAssign<F64> +
    Sub<F64, Output=Self> +
    SubAssign<F64> +
    Div<F64, Output=Self> +
    DivAssign<F64> +
    Rem<F64, Output=Self> +
    RemAssign<F64>
{
    fn constant(constant: f64) -> Self;
    fn to_constant(&self) -> f64;
    // fn ad_num_type() -> ADNumType;
    fn add_scalar(arg1: f64, arg2: Self) -> Self;
    fn sub_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn sub_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn mul_scalar(arg1: f64, arg2: Self) -> Self;
    fn div_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn div_r_scalar(arg1: Self, arg2: f64) -> Self;
    fn rem_l_scalar(arg1: f64, arg2: Self) -> Self;
    fn rem_r_scalar(arg1: Self, arg2: f64) -> Self;
}

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

////////////////////////////////////////////////////////////////////////////////////////////////////

impl AD for f64 {
        fn constant(v: f64) -> Self {
            return v;
        }

        fn to_constant(&self) -> f64 {
            *self
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

/*
pub struct ADAnyNalgebraDMatrix {
    pub (crate) f64: DMatrix<f64>,
    pub (crate) f32: DMatrix<f32>
}
impl ADAnyNalgebraDMatrix {
    pub fn new<T: AD>(item: &DMatrix<T>) -> Self {
        let shape = item.shape();
        let mut f64 = DMatrix::<f64>::zeros(shape.0, shape.1);
        let mut f32 = DMatrix::<f32>::zeros(shape.0, shape.1);

        let s = item.as_slice();
        for (i, ss) in s.iter().enumerate() {
            f64.as_mut_slice()[i] = ss.to_constant();
            f32.as_mut_slice()[i] = ss.to_constant() as f32;
        }

        Self {
            f64, f32
        }
    }
}

#[macro_export]
macro_rules! ad_setup_any_nalgebra_dmatrix {
    ($( $T: ident ),*) => {
        $(
            impl Mul<ADAnyNalgebraDMatrix> for DMatrix<$T> {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: ADAnyNalgebraDMatrix) -> Self::Output {
                    return &self * &rhs.$T
                }
            }

            impl Mul<DMatrix<$T>> for ADAnyNalgebraDMatrix {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: DMatrix<$T>) -> Self::Output {
                    &self.$T * &rhs
                }
            }

            impl Mul<&ADAnyNalgebraDMatrix> for DMatrix<$T> {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: &ADAnyNalgebraDMatrix) -> Self::Output {
                    return &self * &rhs.$T
                }
            }

            impl Mul<DMatrix<$T>> for &ADAnyNalgebraDMatrix {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: DMatrix<$T>) -> Self::Output {
                    &self.$T * &rhs
                }
            }

            impl Mul<ADAnyNalgebraDMatrix> for &DMatrix<$T> {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: ADAnyNalgebraDMatrix) -> Self::Output {
                    return self * &rhs.$T
                }
            }

            impl Mul<&DMatrix<$T>> for ADAnyNalgebraDMatrix {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: &DMatrix<$T>) -> Self::Output {
                    &self.$T * rhs
                }
            }

            impl Mul<&ADAnyNalgebraDMatrix> for &DMatrix<$T> {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: &ADAnyNalgebraDMatrix) -> Self::Output {
                    return self * &rhs.$T
                }
            }

            impl Mul<&DMatrix<$T>> for &ADAnyNalgebraDMatrix {
                type Output = DMatrix<$T>;

                fn mul(self, rhs: &DMatrix<$T>) -> Self::Output {
                    &self.$T * rhs
                }
            }

            impl Mul<DVector<$T>> for ADAnyNalgebraDMatrix {
                type Output = DVector<$T>;

                fn mul(self, rhs: DVector<$T>) -> Self::Output {
                    &self.$T * &rhs
                }
            }

            impl Mul<DVector<$T>> for &ADAnyNalgebraDMatrix {
                type Output = DVector<$T>;

                fn mul(self, rhs: DVector<$T>) -> Self::Output {
                    &self.$T * &rhs
                }
            }

            impl Mul<&DVector<$T>> for ADAnyNalgebraDMatrix {
                type Output = DVector<$T>;

                fn mul(self, rhs: &DVector<$T>) -> Self::Output {
                    &self.$T * rhs
                }
            }

            impl Mul<&DVector<$T>> for &ADAnyNalgebraDMatrix {
                type Output = DVector<$T>;

                fn mul(self, rhs: &DVector<$T>) -> Self::Output {
                    &self.$T * rhs
                }
            }
        )*
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
impl Mul<ADAnyNalgebraDMatrix> for DMatrix<f64> {
    type Output = DMatrix<f64>;

    fn mul(self, rhs: ADAnyNalgebraDMatrix) -> Self::Output {
        return &self * &rhs.f64
    }
}
impl Mul<ADAnyNalgebraDMatrix> for DMatrix<f32> {
    type Output = DMatrix<f32>;

    fn mul(self, rhs: ADAnyNalgebraDMatrix) -> Self::Output {
        return &self * &rhs.f32
    }
}

impl Mul<DMatrix<f64>> for ADAnyNalgebraDMatrix {
    type Output = DMatrix<f64>;

    fn mul(self, rhs: DMatrix<f64>) -> Self::Output {
        &self.f64 * &rhs
    }
}
*/
/*
// pre deprecation
////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug, Copy)]
pub struct NalgebraF64Mat <'a, R: Dim + Clone, C: Dim + Clone, S: Clone + RawStorageMut<f64, R, C>> (pub &'a Matrix<f64, R, C, S>);

impl<
    'a,
    T: AD,
    R1: Dim + Clone,
    C1: Dim + Clone,
    S1: Clone + RawStorageMut<f64, R1, C1>,
    R2: Dim + Clone,
    C2: Dim + Clone,
    S2: Clone + RawStorageMut<T, R2, C2>
>
Mul<&'a Matrix<T, R2, C2, S2>>
for NalgebraF64Mat<'_, R1, C1, S1>
where
    DefaultAllocator: Allocator<T, R2, C2>,
    DefaultAllocator: Allocator<T, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>
{
    type Output = OMatrix<T, R1, C2>;

    fn mul(self, rhs: &Matrix<T, R2, C2, S2>) -> Self::Output {
        mul_nalgebra_f64_matrix_by_nalgebra_matrix(self.0, rhs)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

fn mul_nalgebra_f64_matrix_by_nalgebra_matrix<
    R1: Dim + Clone,
    C1: Dim + Clone,
    S1: Clone + RawStorageMut<f64, R1, C1>,
    R2: Dim + Clone,
    C2: Dim + Clone,
    S2: Clone + RawStorageMut<T, R2, C2>,
    T: AD>(matrix1: &Matrix<f64, R1, C1, S1>, matrix2: &Matrix<T, R2, C2, S2>) -> OMatrix<T, R1, C2>
where
    DefaultAllocator: Allocator<T, R2, C2>,
    DefaultAllocator: Allocator<T, R1, C2>,
    ShapeConstraint: AreMultipliable<R1, C1, R2, C2>
{
    let (num_rows1, num_cols1) = matrix1.shape();
    let (_, num_cols2) = matrix2.shape();
    let s1g = matrix1.shape_generic();
    let s2g = matrix2.shape_generic();
    let mut res = OMatrix::zeros_generic(s1g.0, s2g.1);

    for c2 in 0..num_cols2 {
        for r1 in 0..num_rows1 {
            for c1 in 0..num_cols1 {
                res[(r1, c2)] += T::mul_scalar(matrix1[(r1, c1)], matrix2[(c1, c2)]);
            }
        }
    }

    return res;
}
*/