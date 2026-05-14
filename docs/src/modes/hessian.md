# Hessian / Second-Order AD

`ad_trait` supports computing second-order derivatives (Hessians) by nesting automatic differentiation types. This is achieved through the `HyperAD` family of types.

> [!NOTE]
> The bracketed value `<N>` (e.g., `adfn<N>`, `HessianAD<N>`) is a **const generic** that specifies the number of **tangent lanes**. For Hessian computation, this typically should match the number of input variables you are differentiating with respect to.

| Mode | Type | Best For | Scaling |
| --- | --- | --- | --- |
| **Forward-over-Forward** | `HyperAD_ADFN<N>` | Few inputs ($N < 20$), low memory. | $O(N^2)$ |
| **Forward-over-Reverse** | `HyperAD_ADR<N>` | Many inputs, few outputs (e.g., loss functions). | $O(N)$ |

## Choosing the Right Mode

Selecting the optimal mode depends primarily on the number of input variables ($N$) and the memory constraints of your application.

### Forward-over-Forward (FoF)
- **When to use**: Use this when you have a small number of inputs. It is the most robust mode and has the lowest memory overhead because it does not require building a computation graph.
- **Efficiency**: The computational cost scales quadratically with the number of inputs ($O(N^2)$). For a function with 10 inputs, it is very fast; for 1000 inputs, it becomes prohibitively slow.
- **Implementation**: Uses `HessianAD<N>`.

### Forward-over-Reverse (FoR)
- **When to use**: Use this for functions with many inputs and a single (or few) outputs, such as a neural network loss function or a complex physics simulation.
- **Efficiency**: This mode is significantly more efficient for large $N$. Because the inner layer is Reverse-mode AD, a single backpropagation through a tangent value can recover an entire row of the Hessian. This allows the total cost to scale linearly with the number of inputs ($O(N)$) for scalar-valued functions.
- **Memory**: Higher memory usage than FoF because it must maintain the reverse-mode computation graph.
- **Implementation**: Uses `HessianAD_FOR<N>`.

## Forward-over-Forward Hessian

This mode uses `HyperAD_ADFN`, which is essentially an `adfn` type where the primary value and tangents are themselves `adfn` types.

### Example

```rust
use ad_trait::AD;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, HessianAD, ToOtherADType};
use ad_trait::hyper_ad::hyper::HyperAD_ADFN;

#[derive(Clone)]
struct MyFunc;
impl<T: AD> DifferentiableFunctionTrait<T> for MyFunc {
    const NAME: &'static str = "MyFunc";
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        vec![ x * x * x ] // f(x) = x^3
    }
    fn num_inputs(&self) -> usize { 1 }
    fn num_outputs(&self) -> usize { 1 }
}

fn main() {
    let inputs = [2.0];
    let func = MyFunc;
    let engine = FunctionEngine::new(
        func.clone(), 
        func.to_other_ad_type::<HyperAD_ADFN<1>>(), 
        HessianAD::<1>::new()
    );

    let (f_res, jacobian_res, hessian_res) = engine.hessian(&inputs);
    
    println!("f(2) = {}", f_res[0]);             // 8.0
    println!("f'(2) = {}", jacobian_res[(0,0)]);  // 12.0
    println!("f''(2) = {}", hessian_res[0][(0,0)]); // 12.0
}
```

## Forward-over-Reverse Hessian

This mode uses `HyperAD_ADR`, which uses `adr` as the inner type. This is useful when you want to combine the benefits of forward and reverse mode.

### Example

```rust
use ad_trait::AD;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, HessianAD_FOR, ToOtherADType};
use ad_trait::hyper_ad::hyper_adr::HyperAD_ADR;

#[derive(Clone)]
struct MyFunc;
impl<T: AD> DifferentiableFunctionTrait<T> for MyFunc {
    const NAME: &'static str = "MyFunc";
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        let x = inputs[0];
        vec![ x * x * x ]
    }
    fn num_inputs(&self) -> usize { 1 }
    fn num_outputs(&self) -> usize { 1 }
}

fn main() {
    let inputs = [2.0];
    let func = MyFunc;
    let engine = FunctionEngine::new(
        func.clone(), 
        func.to_other_ad_type::<HyperAD_ADR<1>>(), 
        HessianAD_FOR::<1>::new()
    );

    let (f_res, jacobian_res, hessian_res) = engine.hessian(&inputs);
    
    println!("f'(2) = {}", jacobian_res[(0,0)]);    // 12.0
    println!("f''(2) = {}", hessian_res[0][(0,0)]); // 12.0
}
```

## First-Order Derivatives from Hessian Engines

It is important to note that a `FunctionEngine` initialized for Hessian computation can still be used for standard first-order derivatives. 

Calling `engine.derivative(&inputs)` on a Hessian-enabled block will return the Jacobian matrix as usual. This is possible because the hyper-dual types used for second-order differentiation internally track the first-order gradients as their primal "tangent" values. 

This allows you to maintain a single `FunctionEngine` instance for both standard gradient-based optimization and second-order methods.
