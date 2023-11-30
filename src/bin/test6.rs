use std::borrow::Cow;
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock2;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait2, FiniteDifferencing2};

pub struct TestClass;
impl TestClass {
    pub fn new() -> Self {
        Self {}
    }
}
impl DifferentiableFunctionClass for TestClass {
    type FunctionType<'a, T: AD> = Test;
}

#[derive(Clone)]
pub struct Test;
impl Test {
    pub fn new() -> Self {
        Self {}
    }
}
impl<'a, T: AD> DifferentiableFunctionTrait2<'a, T> for Test {
    fn type_string(&self) -> String {
        "Test".to_string()
    }

    fn call(&self, inputs: &[T]) -> Vec<T> {
        vec![inputs[0].sin()]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

pub struct Test2Class;
impl Test2Class {
    pub fn new() -> Self {
        Self {}
    }
}
impl DifferentiableFunctionClass for Test2Class {
    type FunctionType<'a, T: AD> = Test2<'a, T>;
}

#[derive(Clone)]
pub struct Test2<'a, T: AD> {
    a: &'a T
}
impl<'a, T: AD> Test2<'a, T> {
    pub fn new(a: &'a T) -> Self {
        Self { a }
    }
}
impl<'a, T: AD> DifferentiableFunctionTrait2<'a, T> for Test2<'a, T> {
    fn type_string(&self) -> String {
        "Test2".to_string()
    }

    fn call(&self, inputs: &[T]) -> Vec<T> {
        vec![ inputs[0].sin() * *self.a ]
    }

    fn num_inputs(&self) -> usize {
        1
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

fn main() {
    let dd = DifferentiableBlock2::new(TestClass, FiniteDifferencing2::new(), Cow::Owned(Test), Cow::Owned(Test));
    let res = dd.derivative(&[1.0]);
    println!("{:?}", res);
}