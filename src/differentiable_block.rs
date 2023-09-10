use nalgebra::DMatrix;
use crate::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionTrait};

pub struct DifferentiableBlock<D: DifferentiableFunctionTrait, E: DerivativeMethodTrait> {
    function_standard_args: D::ArgsType<f64>,
    function_derivative_args: D::ArgsType<E::T>,
    derivative_method_data: E::DerivativeMethodData
}
impl<D: DifferentiableFunctionTrait, E: DerivativeMethodTrait> DifferentiableBlock<D, E> {
    pub fn new(function_standard_args: D::ArgsType<f64>,
               function_derivative_args: D::ArgsType<E::T>,
               derivative_method_data: E::DerivativeMethodData) -> Self {
        Self {
            function_standard_args,
            function_derivative_args,
            derivative_method_data,
        }
    }

    pub fn call(&self, inputs: &[f64]) -> Vec<f64> {
        D::call(inputs, &self.function_standard_args)
    }

    pub fn derivative(&self, inputs: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        D::derivative::<E>(inputs, &self.function_derivative_args, &self.derivative_method_data)
    }

    pub fn update_args<U: Fn(&mut D::ArgsType<f64>, &mut D::ArgsType<E::T>) >(&mut self, update_fn: U) {
        (update_fn)(&mut self.function_standard_args, &mut self.function_derivative_args)
    }

    pub fn function_standard_args(&self) -> &D::ArgsType<f64> {
        &self.function_standard_args
    }

    pub fn function_derivative_args(&self) -> &D::ArgsType<E::T> {
        &self.function_derivative_args
    }

    pub fn derivative_method_data(&self) -> &E::DerivativeMethodData {
        &self.derivative_method_data
    }
}