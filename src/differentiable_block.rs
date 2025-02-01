use nalgebra::DMatrix;
use crate::differentiable_function::{DerivativeMethodTrait, DifferentiableFunctionClass, DifferentiableFunctionClassZero, DifferentiableFunctionTrait, DifferentiableFunctionZero};

/*
pub struct DifferentiableBlock<'a, D: DifferentiableFunctionTrait, E: DerivativeMethodTrait, AP: DifferentiableBlockArgPrepTrait<'a, D, E>> {
    function_standard_args: D::ArgsType<'a, f64>,
    function_derivative_args: D::ArgsType<'a, E::T>,
    derivative_method_data: E::DerivativeMethodData,
    args_prep: AP
}
impl<'a, D: DifferentiableFunctionTrait, E: DerivativeMethodTrait, AP: DifferentiableBlockArgPrepTrait<'a, D, E>> DifferentiableBlock<'a, D, E, AP> {
    pub fn new(function_standard_args: D::ArgsType<'a, f64>,
               function_derivative_args: D::ArgsType<'a, E::T>,
               derivative_method_data: E::DerivativeMethodData,
               args_prep: AP) -> Self {
        Self {
            function_standard_args,
            function_derivative_args,
            derivative_method_data,
            args_prep,
        }
    }

    pub fn call(&self, inputs: &[f64]) -> Vec<f64> {
        self.prep_args(inputs);
        D::call(inputs, &self.function_standard_args)
    }

    pub fn derivative(&self, inputs: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        self.prep_args(inputs);
        D::derivative::<E>(inputs, &self.function_derivative_args, &self.derivative_method_data)
    }

    pub (crate) fn prep_args(&self, inputs: &[f64]) {
        // AP::prep_args(inputs, &mut self.function_standard_args, &mut self.function_derivative_args);
        self.args_prep.prep_args(inputs, &self.function_standard_args, &self.function_derivative_args);
    }

    pub fn update_args<U: Fn(&mut D::ArgsType<'_, f64>, &mut D::ArgsType<'_, E::T>) >(&mut self, update_fn: U) {
        (update_fn)(&mut self.function_standard_args, &mut self.function_derivative_args)
    }

    /*
    pub fn update_args2<A: DifferentiableBlockUpdateArgs<'a, D>>(&mut self, u: A) {
        u.update_args(&mut self.function_standard_args);
        u.update_args(&mut self.function_derivative_args);
    }
    */

    pub fn function_standard_args(&self) -> &D::ArgsType<'a, f64> {
        &self.function_standard_args
    }

    pub fn function_derivative_args(&self) -> &D::ArgsType<'a, E::T> {
        &self.function_derivative_args
    }

    pub fn derivative_method_data(&self) -> &E::DerivativeMethodData {
        &self.derivative_method_data
    }
}
*/

/*
/// Has to be separate so that both the standard args and derivative args can be updated at the same time
pub trait DifferentiableBlockArgPrepTrait<'a, D: DifferentiableFunctionTrait, E: DerivativeMethodTrait> {
    fn prep_args(&self, inputs: &[f64], function_standard_args: &D::ArgsType<'a, f64>, function_derivative_args: &D::ArgsType<'a, E::T>);
}
impl<'a, D: DifferentiableFunctionTrait, E: DerivativeMethodTrait> DifferentiableBlockArgPrepTrait<'a, D, E> for () {
    fn prep_args(&self, _inputs: &[f64], _function_standard_args: &D::ArgsType<'a, f64>, _function_derivative_args: &D::ArgsType<'a, E::T>) { }
}
*/

/*
pub struct DifferentiableBlock2<'a, E: DerivativeMethodTrait2> {
    function_standard: Box<dyn DifferentiableFunctionTrait2<'a, f64> + 'a>,
    function_derivative: Box<dyn DifferentiableFunctionTrait2<'a, E::T> + 'a>,
    derivative_method: E
}
impl<'a, E: DerivativeMethodTrait2> DifferentiableBlock2<'a, E> {
    pub fn new<D1, D2>(derivative_method: E, function_standard: D1, function_derivative: D2) -> Self
        where D1: DifferentiableFunctionTrait2<'a, f64> + 'a,
              D2: DifferentiableFunctionTrait2<'a, E::T> + 'a {
        assert_eq!(function_standard.type_string(), function_derivative.type_string(), "must be the same type");

        Self {
            function_standard: Box::new(function_standard),
            function_derivative: Box::new(function_derivative),
            derivative_method,
        }
    }

    #[inline]
    pub fn call(&self, inputs: &[f64]) -> Vec<f64> {
        self.function_standard.call(inputs)
    }

    #[inline]
    pub fn derivative(&self, inputs: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        self.derivative_method.derivative(inputs, &*self.function_derivative)
    }

    pub fn update_args<DC: DifferentiableFunctionClass + 'static, U: Fn(&mut DC::FunctionType<'a, f64>, &mut DC::FunctionType<'a, E::T>) >(&'a mut self, update_fn: U) {
        // let mut fs = self.function_standard.as_mut().

        // (update_fn)(&mut self.function_standard_args, &mut self.function_derivative_args)

    }
}
*/

/*
pub struct DifferentiableBlock2<'a, E: DerivativeMethodTrait2, D1: DifferentiableFunctionTrait2<'a, f64>, D2: DifferentiableFunctionTrait2<'a, E::T>> {
    function_standard: D1,
    function_derivative: D2,
    derivative_method: E,
    phantom_data: PhantomData<&'a ()>
}
impl<'a, E: DerivativeMethodTrait2, D1: DifferentiableFunctionTrait2<'a, f64>, D2: DifferentiableFunctionTrait2<'a, E::T>> DifferentiableBlock2<'a, E, D1, D2> {
    pub fn new(function_standard: D1, function_derivative: D2, derivative_method: E) -> Self {
        assert_eq!(function_derivative.type_string(), function_standard.type_string(), "differentiable block must have functions of the same type");
        Self { function_standard, function_derivative, derivative_method, phantom_data: Default::default() }
    }

    #[inline]
    pub fn call(&self, inputs: &[f64]) -> Vec<f64> {
        self.function_standard.call(inputs)
    }

    #[inline]
    pub fn derivative(&self, inputs: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        self.derivative_method.derivative(inputs, &self.function_derivative)
    }

    pub fn update_function<U: Fn(&D1, &D2) >(&'a self, update_fn: U) {
        update_fn(&self.function_standard, &self.function_derivative)
    }
}
*/


#[derive(Clone)]
pub struct DifferentiableBlock<DC: DifferentiableFunctionClass, E: DerivativeMethodTrait> {
    function_standard: DC::FunctionType<f64>,
    function_derivative: DC::FunctionType<E::T>,
    derivative_method: E
}
impl<DC: DifferentiableFunctionClass, E: DerivativeMethodTrait> DifferentiableBlock<DC, E> {
    pub fn new(derivative_method: E, function_standard: DC::FunctionType<f64>, function_derivative: DC::FunctionType<E::T>) -> Self {
        Self {
            function_standard,
            function_derivative,
            derivative_method,
        }
    }
    pub fn new_with_tag(_differentiable_function_class: DC, derivative_method: E, function_standard: DC::FunctionType<f64>, function_derivative: DC::FunctionType<E::T>) -> Self {
        Self {
            function_standard,
            function_derivative,
            derivative_method,
        }
    }
    #[inline(always)]
    pub fn num_inputs(&self) -> usize {
        self.function_standard.num_inputs()
    }
    #[inline(always)]
    pub fn num_outputs(&self) -> usize {
        self.function_standard.num_outputs()
    }
    #[inline]
    pub fn call(&self, inputs: &[f64]) -> Vec<f64> {
        self.function_standard.call(inputs, false)
    }
    #[inline]
    pub fn call_ad_values(&self, inputs: &[E::T]) -> Vec<E::T> {
        self.function_derivative.call(inputs, false)
    }
    #[inline]
    pub fn derivative(&self, inputs: &[f64]) -> (Vec<f64>, DMatrix<f64>) {
        self.derivative_method.derivative(inputs, &self.function_derivative)
    }
    pub fn update_function<U: Fn(&DC::FunctionType<f64>, &DC::FunctionType<E::T>) >(&self, update_fn: U) {
        update_fn(&self.function_standard, &self.function_derivative)
    }
    #[inline]
    pub fn function_standard(&self) -> &DC::FunctionType<f64> {
        &self.function_standard
    }
    #[inline]
    pub fn function_derivative(&self) -> &DC::FunctionType<E::T> {
        &self.function_derivative
    }
    #[inline]
    pub fn derivative_method(&self) -> &E {
        &self.derivative_method
    }
}
pub type DifferentiableBlockEmpty = DifferentiableBlock<(), ()>;
impl DifferentiableBlockEmpty {
    pub fn new_empty() -> Self  {
        Self::new((), (), ())
    }
}

pub type DifferentiableBlockZero<E> = DifferentiableBlock<DifferentiableFunctionClassZero, E>;
impl<E: DerivativeMethodTrait> DifferentiableBlockZero<E> {
    pub fn new_zero(derivative_method: E, num_inputs: usize, num_outputs: usize) -> Self {
        Self::new(derivative_method, DifferentiableFunctionZero::new(num_inputs, num_outputs), DifferentiableFunctionZero::new(num_inputs, num_outputs))
    }
}

/*
#[allow(dead_code)]
pub struct DifferentiableBlock2<'a, E: DerivativeMethodTrait> {
    functions_standard: Vec<Box<dyn DifferentiableFunctionTrait2<'a, f64>>>,
    functions_derivative: Vec<Box<dyn DifferentiableFunctionTrait2<'a, E::T>>>,
    weights_standard: Vec<f64>,
    weights_derivative: Vec<E::T>,
    derivative_method: E
}
#[allow(unused_variables)]
impl<'a, E: DerivativeMethodTrait> DifferentiableBlock2<'a, E> {
    pub fn new_empty(derivative_method: E) -> Self {
        Self {
            functions_standard: vec![],
            functions_derivative: vec![],
            weights_standard: vec![],
            weights_derivative: vec![],
            derivative_method,
        }
    }
    pub fn insert_function<F1: DifferentiableFunctionTrait2<'a, f64> + 'a, F2: DifferentiableFunctionTrait2<'a, E::T> + 'a>(mut self, function_standard: F1, function_derivative: F2, weight: f64) -> Self {
        assert_eq!(function_standard.name(), function_derivative.name());

        self.functions_standard.push(Box::new(function_standard));
        self.functions_derivative.push(Box::new(function_derivative));
        self.weights_standard.push(weight);
        self.weights_derivative.push(weight.to_other_ad_type::<E::T>());

        self
    }
    pub fn update_weight(&mut self, function_idx: usize, weight: f64) {
        self.weights_standard[function_idx] = weight;
        self.weights_derivative[function_idx] = weight.to_other_ad_type::<E::T>();
    }
    pub fn update_function<F1: DifferentiableFunctionTrait2<'a, f64>, F2: DifferentiableFunctionTrait2<'a, E::T>>(&self, function_idx: usize) {
        // self.functions_standard[function_idx].downcast_ref::<F1>();

        todo!()
    }
}
*/

/*
pub trait DifferentiableBlockArgPrepTrait2<'a, DC: DifferentiableFunctionClass, E: DerivativeMethodTrait2> {
    fn prep_args(&self, inputs: &[f64], function_standard_args: &DC::FunctionType<'a, f64>, function_derivative_args: &DC::FunctionType<'a, E::T>);
}
impl<'a, DC: DifferentiableFunctionClass, E: DerivativeMethodTrait2> DifferentiableBlockArgPrepTrait2<'a, DC, E> for () {
    fn prep_args(&self, _inputs: &[f64], _function_standard_args: &DC::FunctionType<'a, f64>, _function_derivative_args: &DC::FunctionType<'a, E::T>) { }
}
*/
/*
pub trait DifferentiableBlockUpdateArgs<'a, D: DifferentiableFunctionTrait> {
    fn update_args<T: AD>(&self, args: &mut D::ArgsType<'_, T>);
}
*/