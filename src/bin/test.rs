use ad_trait::AD;
use ad_trait::reverse_ad::adr::adr;

#[cfg(feature = "nightly")]
use ad_trait::forward_ad::adf::adf_f64x8;
#[cfg(feature = "nightly")]
use ad_trait::forward_ad::ForwardADTrait;

#[cfg(feature = "hessian")]
#[derive(Clone)]
struct RosenbrockFunction;

#[cfg(feature = "hessian")]
impl<T: AD> ad_trait::differentiable_function::DifferentiableFunctionTrait<T> for RosenbrockFunction {
    const NAME: &'static str = "RosenbrockFunction";

    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let mut sum = T::constant(0.0);
        for i in 0..(inputs.len() - 1) {
            let x_i = inputs[i];
            let x_next = inputs[i + 1];
            let term1 = x_next - x_i * x_i;
            let term2 = T::constant(1.0) - x_i;
            sum = sum + T::constant(100.0) * term1 * term1 + term2 * term2;
        }
        vec![sum]
    }

    fn num_inputs(&self) -> usize {
        10
    }

    fn num_outputs(&self) -> usize {
        1
    }
}

#[cfg(feature = "hessian")]
impl ad_trait::differentiable_function::Reparameterize for RosenbrockFunction {
    type SelfType<T2: AD> = RosenbrockFunction;
}

fn rosenbrock<T: AD>(inputs: &[T]) -> T {
    let mut sum = T::constant(0.0);
    for i in 0..(inputs.len() - 1) {
        let x_i = inputs[i];
        let x_next = inputs[i + 1];
        let term1 = x_next - x_i * x_i;
        let term2 = T::constant(1.0) - x_i;
        sum = sum + T::constant(100.0) * term1 * term1 + term2 * term2;
    }
    sum
}

fn main() {
    println!("=== Correctness Runner ===");

    // Define input data
    let rosenbrock_inputs = vec![1.2, 1.5, 0.8, 2.0, 1.1, 0.5, 1.7, 0.9, 1.3, 1.6];

    // Check correctness for Reverse Mode AD
    let mut inputs_ad = vec![];
    for (i, &input) in rosenbrock_inputs.iter().enumerate() {
        inputs_ad.push(adr::new_variable(input, i == 0));
    }
    let res = rosenbrock(&inputs_ad);
    let grad_output = res.get_backwards_mode_grad();
    let grads: Vec<f64> = inputs_ad.iter().map(|x| grad_output.wrt(x)).collect();

    assert!((res.value() - 1920.24).abs() < 1e-2, "Rosenbrock value mismatch!");
    assert!((grads[0] - (-28.4)).abs() < 1e-2, "Rosenbrock gradient mismatch at index 0!");
    println!("Reverse AD Correctness: PASSED");

    #[cfg(feature = "nightly")]
    {
        let mut vars: Vec<adf_f64x8> = rosenbrock_inputs.iter().map(|&x| adf_f64x8::constant(x)).collect();
        for i in 0..8 {
            vars[i].set_tangent_value(i, 1.0);
        }
        let res_f = rosenbrock(&vars);
        assert!((res_f.value() - 1920.24).abs() < 1e-2, "Forward AD value mismatch!");
        println!("Forward AD Correctness: PASSED");
    }

    #[cfg(feature = "hessian")]
    {
        use ad_trait::differentiable_function::HessianAD_FOR;
        use ad_trait::function_engine::FunctionEngine;

        let func = RosenbrockFunction;
        let engine = FunctionEngine::new(func.clone(), func, HessianAD_FOR::<2>::new());
        let (val, grad, hess) = engine.hessian(&rosenbrock_inputs);

        assert!((val[0] - 1920.24).abs() < 1e-2, "Hessian value mismatch!");
        assert!((grad[(0, 0)] - (-28.4)).abs() < 1e-2, "Hessian gradient mismatch at index 0!");
        assert!((hess[0][(0, 0)] - 1130.0).abs() < 1e-2, "Hessian value mismatch at index 0,0!");
        println!("Hessian AD Correctness: PASSED");
    }
}