use ad_trait::differentiable_function::{
    DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, Reparameterize,
    ReverseAD,
};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::AD;
use approx::assert_relative_eq;

#[derive(Clone)]
struct PolynomialTest;

impl<T: AD> DifferentiableFunctionTrait<T> for PolynomialTest {
    const NAME: &'static str = "PolynomialTest";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        // f(x) = 2x^2 + 3x + 1
        // f'(x) = 4x + 3
        vec![x * x * T::constant(2.0) + x * T::constant(3.0) + T::constant(1.0)]
    }

    fn num_inputs(&self) -> usize {
        1
    }
    fn num_outputs(&self) -> usize {
        1
    }
}

impl Reparameterize for PolynomialTest {
    type SelfType<T2: AD> = PolynomialTest;
}

#[derive(Clone)]
struct MultiVariateTest;

impl<T: AD> DifferentiableFunctionTrait<T> for MultiVariateTest {
    const NAME: &'static str = "MultiVariateTest";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        let y = inputs[1];
        // f(x, y) = [x^2 + y, y^2 + x]
        // J = [[2x, 1],
        //      [1, 2y]]
        vec![x * x + y, y * y + x]
    }

    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
}

impl Reparameterize for MultiVariateTest {
    type SelfType<T2: AD> = MultiVariateTest;
}

#[derive(Clone)]
struct MatrixMulTest;

impl<T: AD> DifferentiableFunctionTrait<T> for MatrixMulTest {
    const NAME: &'static str = "MatrixMulTest";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        let y = inputs[1];

        let m = nalgebra::Matrix2::new(x, y, T::constant(0.0), x * y);
        let v = nalgebra::Vector2::new(T::constant(1.0), T::constant(2.0));
        let res = m * v;

        // res = [x + 2y, 2xy]
        // d res[0] / dx = 1
        // d res[0] / dy = 2
        // d res[1] / dx = 2y
        // d res[1] / dy = 2x
        vec![res[0], res[1]]
    }

    fn num_inputs(&self) -> usize {
        2
    }
    fn num_outputs(&self) -> usize {
        2
    }
}

impl Reparameterize for MatrixMulTest {
    type SelfType<T2: AD> = MatrixMulTest;
}

#[test]
fn test_polynomial_forward_ad() {
    let func = PolynomialTest;
    let engine = FunctionEngine::new(func.clone(), func, ForwardAD::new());

    let x = 2.0;
    let (val, grad) = engine.derivative(&[x]);

    // f(2) = 2(2^2) + 3(2) + 1 = 8 + 6 + 1 = 15
    // f'(2) = 4(2) + 3 = 11
    assert_relative_eq!(val[0], 15.0);
    assert_relative_eq!(grad[(0, 0)], 11.0);
}

#[test]
fn test_polynomial_reverse_ad() {
    let func = PolynomialTest;
    let engine = FunctionEngine::new(func.clone(), func, ReverseAD::new());

    let x = 2.0;
    let (val, grad) = engine.derivative(&[x]);

    assert_relative_eq!(val[0], 15.0);
    assert_relative_eq!(grad[(0, 0)], 11.0);
}

#[test]
fn test_polynomial_finite_differencing() {
    let func = PolynomialTest;
    let engine = FunctionEngine::new(func.clone(), func, FiniteDifferencing::new());

    let x = 2.0;
    let (val, grad) = engine.derivative(&[x]);

    assert_relative_eq!(val[0], 15.0);
    assert_relative_eq!(grad[(0, 0)], 11.0, epsilon = 1e-5);
}

#[test]
fn test_multivariate_forward_ad() {
    let func = MultiVariateTest;
    let engine = FunctionEngine::new(func.clone(), func, ForwardAD::new());

    let inputs = [2.0, 3.0];
    let (val, grad) = engine.derivative(&inputs);

    // f(2, 3) = [2^2 + 3, 3^2 + 2] = [7, 11]
    // J = [[2x, 1], [1, 2y]] = [[4, 1], [1, 6]]
    assert_relative_eq!(val[0], 7.0);
    assert_relative_eq!(val[1], 11.0);

    assert_relative_eq!(grad[(0, 0)], 4.0);
    assert_relative_eq!(grad[(0, 1)], 1.0);
    assert_relative_eq!(grad[(1, 0)], 1.0);
    assert_relative_eq!(grad[(1, 1)], 6.0);
}

#[test]
fn test_multivariate_reverse_ad() {
    let func = MultiVariateTest;
    let engine = FunctionEngine::new(func.clone(), func, ReverseAD::new());

    let inputs = [2.0, 3.0];
    let (val, grad) = engine.derivative(&inputs);

    assert_relative_eq!(val[0], 7.0);
    assert_relative_eq!(val[1], 11.0);

    assert_relative_eq!(grad[(0, 0)], 4.0);
    assert_relative_eq!(grad[(0, 1)], 1.0);
    assert_relative_eq!(grad[(1, 0)], 1.0);
    assert_relative_eq!(grad[(1, 1)], 6.0);
}

#[test]
fn test_multivariate_forward_ad_multi() {
    let func = MultiVariateTest;
    let engine = FunctionEngine::new(func.clone(), func, ForwardADMulti::<adfn<2>>::new());

    let inputs = [2.0, 3.0];
    let (val, grad) = engine.derivative(&inputs);

    assert_relative_eq!(val[0], 7.0);
    assert_relative_eq!(val[1], 11.0);

    assert_relative_eq!(grad[(0, 0)], 4.0);
    assert_relative_eq!(grad[(0, 1)], 1.0);
    assert_relative_eq!(grad[(1, 0)], 1.0);
    assert_relative_eq!(grad[(1, 1)], 6.0);
}

#[test]
fn test_matrix_mul_forward_ad() {
    let func = MatrixMulTest;
    let engine = FunctionEngine::new(func.clone(), func, ForwardAD::new());

    let inputs = [2.0, 3.0];
    let (val, grad) = engine.derivative(&inputs);

    // res = [x + 2y, 2xy] = [2 + 6, 2*2*3] = [8, 12]
    assert_relative_eq!(val[0], 8.0);
    assert_relative_eq!(val[1], 12.0);

    // J = [[1, 2], [2y, 2x]] = [[1, 2], [6, 4]]
    assert_relative_eq!(grad[(0, 0)], 1.0);
    assert_relative_eq!(grad[(0, 1)], 2.0);
    assert_relative_eq!(grad[(1, 0)], 6.0);
    assert_relative_eq!(grad[(1, 1)], 4.0);
}

#[test]
fn test_matrix_mul_reverse_ad() {
    let func = MatrixMulTest;
    let engine = FunctionEngine::new(func.clone(), func, ReverseAD::new());

    let inputs = [2.0, 3.0];
    let (val, grad) = engine.derivative(&inputs);

    assert_relative_eq!(val[0], 8.0);
    assert_relative_eq!(val[1], 12.0);

    assert_relative_eq!(grad[(0, 0)], 1.0);
    assert_relative_eq!(grad[(0, 1)], 2.0);
    assert_relative_eq!(grad[(1, 0)], 6.0);
    assert_relative_eq!(grad[(1, 1)], 4.0);
}
