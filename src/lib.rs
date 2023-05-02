use std::fmt::{Debug, Display};
use nalgebra::{Dim, Matrix, RawStorageMut};
use simba::scalar::ComplexField;

pub trait AD :
    ComplexField +
    PartialOrd +
    PartialEq +
    Debug +
    Display
{
    fn constant(constant: f64) -> Self;
    fn scalar_add(arg1: f64, arg2: Self) -> Self;
    fn scalar_lsub(arg1: f64, arg2: Self) -> Self;
    fn scalar_rsub(arg1: Self, arg2: f64) -> Self;
    fn scalar_mul(arg1: f64, arg2: Self) -> Self;
    fn scalar_ldiv(arg1: f64, arg2: Self) -> Self;
    fn scalar_rdiv(arg1: Self, arg2: f64) -> Self;
    fn scalar_lrem(arg1: f64, arg2: Self) -> Self;
    fn scalar_rrem(arg1: Self, arg2: f64) -> Self;
    fn scalar_mul_by_nalgebra_matrix<R: Dim + Clone, C: Dim + Clone, S: Clone + RawStorageMut<Self, R, C>>(scalar: f64, matrix: &Matrix<Self, R, C, S> ) -> Matrix<Self, R, C, S>;
}

impl AD for f64 {
    fn constant(v: f64) -> Self {
        return v;
    }

    fn scalar_add(arg1: f64, arg2: Self) -> Self {
        arg1 + arg2
    }

    fn scalar_lsub(arg1: f64, arg2: Self) -> Self {
        arg1 - arg2
    }

    fn scalar_rsub(arg1: Self, arg2: f64) -> Self {
        arg1 - arg2
    }

    fn scalar_mul(arg1: f64, arg2: Self) -> Self {
        arg1 * arg2
    }

    fn scalar_ldiv(arg1: f64, arg2: Self) -> Self {
        arg1 / arg2
    }

    fn scalar_rdiv(arg1: Self, arg2: f64) -> Self {
        arg1 / arg2
    }

    fn scalar_lrem(arg1: f64, arg2: Self) -> Self {
        arg1 % arg2
    }

    fn scalar_rrem(arg1: Self, arg2: f64) -> Self {
        arg1 % arg2
    }

    fn scalar_mul_by_nalgebra_matrix<R: Dim + Clone, C: Dim + Clone, S: Clone + RawStorageMut<Self, R, C>>(scalar: f64, matrix: &Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = matrix.clone();
        out.iter_mut().for_each(|x| *x*=scalar );
        out
    }
}

impl AD for f32 {
    fn constant(v: f64) -> Self {
        return v as f32;
    }

    fn scalar_add(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 + arg2
    }

    fn scalar_lsub(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 - arg2
    }

    fn scalar_rsub(arg1: Self, arg2: f64) -> Self {
        arg1 - arg2 as f32
    }

    fn scalar_mul(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 * arg2
    }

    fn scalar_ldiv(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 / arg2
    }

    fn scalar_rdiv(arg1: Self, arg2: f64) -> Self {
        arg1 / arg2 as f32
    }

    fn scalar_lrem(arg1: f64, arg2: Self) -> Self {
        arg1 as f32 % arg2
    }

    fn scalar_rrem(arg1: Self, arg2: f64) -> Self {
        arg1 % arg2 as f32
    }

    fn scalar_mul_by_nalgebra_matrix<R: Dim + Clone, C: Dim + Clone, S: Clone + RawStorageMut<Self, R, C>>(scalar: f64, matrix: &Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = matrix.clone();
        out.iter_mut().for_each(|x| *x*=scalar as f32 );
        out
    }
}