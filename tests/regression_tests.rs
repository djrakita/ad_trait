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

        let m = ad_trait::nalgebra::Matrix2::new(x, y, T::constant(0.0), x * y);
        let v = ad_trait::nalgebra::Vector2::new(T::constant(1.0), T::constant(2.0));
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

#[cfg(feature = "hessian")]
#[test]
fn test_scalar_hessian() {
    use ad_trait::AD;
    use ad_trait::forward_ad::ForwardADTrait;
    use ad_trait::forward_ad::adfn::adfn;
    use ad_trait::hyper_ad::hyper::HyperAD_ADFN;

    // We want to differentiate f(x) = x^3 + 2x^2 + 5x + 1
    // f'(x) = 3x^2 + 4x + 5
    // f''(x) = 6x + 4
    // At x = 2.0:
    // f(2) = 8 + 8 + 10 + 1 = 27
    // f'(2) = 12 + 8 + 5 = 25
    // f''(2) = 12 + 4 = 16

    // The inner type adfn<1> evaluates the first derivative
    // The outer type HyperAD_ADFN<1> evaluates the second derivative

    let mut x_inner = adfn::<1>::constant(2.0);
    x_inner.set_tangent_value(0, 1.0); // inner derivative with respect to x

    let mut x_outer = HyperAD_ADFN::<1>::new_inner_constant(x_inner);
    // Outer tangent gets the inner AD type where its primary value is 1.0 (for derivative with respect to x)
    // and its tangent is 0.0
    x_outer.set_tangent_value(0, 1.0);

    // f(x)
    let y = x_outer * x_outer * x_outer + HyperAD_ADFN::<1>::new_constant(2.0) * x_outer * x_outer + HyperAD_ADFN::<1>::new_constant(5.0) * x_outer + HyperAD_ADFN::<1>::new_constant(1.0);

    // The primal value of the outer type is the evaluation of the inner type (f(x) and f'(x))
    let y_inner = y.inner_value(); // adfn<1>
    let _f_val = y_inner.value();
    let _f_prime = y_inner.tangent_as_vec()[0];

    // The tangent of the outer type holds the derivative of the inner type!
    let y_outer_tangent = y.inner_tangent(); // This returns an array of inner adfn values
    
    // The value of the outer tangent is f'(x) evaluated on the inner AD type,
    // which results in an adfn with primal value f'(2.0) = 25.0, and tangent value f''(2.0) = 16.0.
    let _f_prime_from_outer = y_outer_tangent[0].value();
    let f_double_prime = y_outer_tangent[0].tangent_as_vec()[0];

    assert_eq!(f_double_prime, 16.0);
}

#[cfg(feature = "hessian")]
#[test]
fn test_function_engine_hessian() {
    use ad_trait::differentiable_function::HessianAD;
    let func = PolynomialTest;
    let engine = FunctionEngine::new(func.clone(), func, HessianAD::<1>::new());

    let x = 2.0;
    let (val, grad, hess) = engine.hessian(&[x]);

    // f(2) = 2(2^2) + 3(2) + 1 = 15
    // f'(x) = 4x + 3 => f'(2) = 11
    // f''(x) = 4
    assert_relative_eq!(val[0], 15.0);
    assert_relative_eq!(grad[(0, 0)], 11.0);
    assert_relative_eq!(hess[0][(0, 0)], 4.0);
}

#[cfg(feature = "hessian")]
#[test]
fn test_hessian_batching() {
    use ad_trait::differentiable_function::{HessianAD, HessianAD_FOR};
    let func = MultiVariateTest; // f(x, y) = [x^2 + y, y^2 + x]
    
    // Test Forward-over-Forward batching with N=1 (4 passes)
    let engine_fof = FunctionEngine::new(func.clone(), func.clone(), HessianAD::<1>::new());
    let inputs = [2.0, 3.0];
    let (val, grad, hess) = engine_fof.hessian(&inputs);

    // f(2, 3) = [7, 11]
    // J = [[4, 1], [1, 6]]
    // H0 = [[2, 0], [0, 0]]
    // H1 = [[0, 0], [0, 2]]
    assert_relative_eq!(val[0], 7.0);
    assert_relative_eq!(grad[(0, 0)], 4.0);
    assert_relative_eq!(grad[(0, 1)], 1.0);
    assert_relative_eq!(hess[0][(0, 0)], 2.0);
    assert_relative_eq!(hess[0][(1, 1)], 0.0);
    assert_relative_eq!(hess[1][(1, 1)], 2.0);

    // Test Forward-over-Reverse batching with N=1 (2 passes)
    let engine_for = FunctionEngine::new(func.clone(), func, HessianAD_FOR::<1>::new());
    let (val_for, grad_for, hess_for) = engine_for.hessian(&inputs);

    assert_relative_eq!(val_for[0], 7.0);
    assert_relative_eq!(grad_for[(0, 0)], 4.0);
    assert_relative_eq!(hess_for[0][(0, 0)], 2.0);
    assert_relative_eq!(hess_for[1][(1, 1)], 2.0);
}
