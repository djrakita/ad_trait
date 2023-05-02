use std::fmt::{Debug, Display};
use std::ops::{Add, Div, Mul, Rem, Sub};
use nalgebra::{Dim, Matrix, RawStorageMut};
use simba::scalar::ComplexField;

pub trait AD :
    Add<f64, Output=Self> +
    Mul<f64, Output=Self> +
    Sub<f64, Output=Self> +
    Div<f64, Output=Self> +
    Rem<f64, Output=Self> +
    ComplexField +
    PartialOrd +
    PartialEq +
    Debug +
    Display
{
    fn constant(v: f64) -> Self;
    fn scalar_multiply_by_nalgebra_matrix<R: Dim + Clone, C: Dim + Clone, S: Clone + RawStorageMut<Self, R, C>>( scalar: f64, matrix: &Matrix<Self, R, C, S> ) -> Matrix<Self, R, C, S>;
}


impl AD for f64 {
    fn constant(v: f64) -> Self {
        return v;
    }
    fn scalar_multiply_by_nalgebra_matrix<R: Dim + Clone, C: Dim + Clone, S: Clone + RawStorageMut<Self, R, C>>(scalar: f64, matrix: &Matrix<Self, R, C, S>) -> Matrix<Self, R, C, S> {
        let mut out = matrix.clone();
        out.iter_mut().for_each(|x| *x*=scalar );
        out
    }
}