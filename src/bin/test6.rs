use std::cell::RefCell;
use std::rc::Rc;
use ad_trait::AD;
use ad_trait::differentiable_function::{DifferentiableFunctionClass, DifferentiableFunctionTrait2};

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
    fn call(&self, inputs: &[T], frozen: bool) -> Vec<T> {
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
    fn call(&self, inputs: &[T], frozen: bool) -> Vec<T> {
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
    let a: Rc<RefCell<Option<i32>>> = Rc::new(RefCell::new(None));

    let b = a.clone();
    let c = b.clone();

    *b.borrow_mut() = Some(2);

    println!("{:?}", c);
}