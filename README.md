# ad_trait

[![](https://img.shields.io/crates/v/ad_trait.svg)](https://crates.io/crates/ad_trait) [![](https://img.shields.io/badge/docs-book-blue)](https://djrakita.github.io/ad_trait/) [![](https://docs.rs/ad_trait/badge.svg)](https://docs.rs/ad_trait) [![](https://github.com/djrakita/ad_trait/actions/workflows/deploy_book.yml/badge.svg)](https://github.com/djrakita/ad_trait/actions/workflows/deploy_book.yml) [![](https://github.com/djrakita/ad_trait/actions/workflows/rust.yml/badge.svg)](https://github.com/djrakita/ad_trait/actions/workflows/rust.yml)

## Introduction
This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
Rust programming language. Utilizing Rust's extensive and expressive trait features, the several
types in this crate that implement the trait AD can be thought of as a drop-in replacement for an
f64 or f32 that affords forward mode or backwards mode automatic differentiation on any downstream
computation in Rust.
## Key Features
- ad_trait supports reverse mode or forward mode automatic differentiation. The forward mode automatic
differentiation implementation can also take advantage of SIMD to compute multiple tangents simultaneously.
- **Second-Order AD**: Supports computing Hessians via recursive dual types, including Forward-over-Forward and Forward-over-Reverse modes.
- The core rust f64 or f32 types also implement the AD trait, meaning any functions that take an AD
trait object as a generic type can handle either standard floating point computation or derivative
tracking automatic differentiation with essentially no overhead.
- The provided types that implement the AD trait also implement several useful traits that allow it
to operate almost exactly as a standard f64. For example, it even implements the `RealField` and
`ComplexField` traits, meaning it can be used in any `nalgebra` or `ndarray` computations.

## Example
```rust
use ad_trait::AD;
use ad_trait::function_engine::FunctionEngine;
use ad_trait::differentiable_function::{DifferentiableFunctionTrait, FiniteDifferencing, ForwardAD, ForwardADMulti, ReverseAD};
use ad_trait::forward_ad::adfn::adfn;
use ad_trait::reverse_ad::adr::adr;

#[derive(Clone)]
pub struct Test<T: AD> {
  coeff: T
}
impl<T: AD> DifferentiableFunctionTrait<T> for Test<T> {
  const NAME: &'static str = "Test";

  fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
    vec![ self.coeff*inputs[0].sin() + inputs[1].cos() ]
  }

  fn num_inputs(&self) -> usize {
    2
  }

  fn num_outputs(&self) -> usize {
    1
  }
}
impl<T: AD> Test<T> {
  pub fn to_other_ad_type<T2: AD>(&self) -> Test<T2> {
    Test { coeff: self.coeff.to_other_ad_type::<T2>() }
  }
}


fn main() {
  let inputs = vec![1., 2.];

  // Reverse AD //////////////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.to_other_ad_type::<adr>();
  let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ReverseAD::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Reverse AD: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Forward AD //////////////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.to_other_ad_type::<adfn<1>>();
  let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardAD::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Forward AD: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Forward AD Multi ////////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.to_other_ad_type::<adfn<2>>();
  let differentiable_block = FunctionEngine::new(function_standard, function_derivative, ForwardADMulti::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Forward AD Multi: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Finite Differencing /////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.clone();
  let differentiable_block = FunctionEngine::new(function_standard, function_derivative, FiniteDifferencing::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Finite Differencing: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Second-Order Derivatives (Hessian) //////////////////////////////////////////////////////////
  // Using HessianAD for Forward-over-Forward Hessian
  use ad_trait::differentiable_function::HessianAD;
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.to_other_ad_type::<HyperAD_ADFN<1>>();
  let differentiable_block = FunctionEngine::new(function_standard, function_derivative, HessianAD::<1>::new());

  let (f_res, jacobian_res, hessian_res) = differentiable_block.hessian(&inputs);
  println!("Second-Order AD: ");
  println!("  f_res: {}", f_res[0]);
  println!("  jacobian: {}", jacobian_res);
  println!("  hessian: {:?}", hessian_res);
  println!("//////////////");
  println!();
}
```

## Changelog

### [0.3.1]
- **Performance Optimizations**: Dramatically improved execution speeds for Reverse AD, Forward AD, and Hessian AD (`HessianAD_FOR` mode) without any breaking API changes.
- **Optimized Reverse AD (`adr`)**: Reduced computation graph node size by 55% (from 88 to 40 bytes) and replaced thread-local `RwLock` operations with zero-overhead `RefCell`/`Cell` structures.
- **Optimized Forward AD (`adf`)**: Eliminated heap allocations during SIMD multi-tangent operations in favor of stack arrays, allowing the compiler to autovectorize calculations directly into hardware registers.
- **Accelerated Hessian AD (`HessianAD_FOR`)**: Sped up Forward-over-Reverse Hessian computations by utilizing the optimized reverse AD engine.
- **Stable Forward AD (`adfn`)**: Confirmed that `adfn` runs entirely on stable Rust (no nightly compiler required). Because `adfn` uses stack-allocated arrays `[f64; N]`, it was already zero-allocation and did not require changes.

#### Performance Benchmarks (Macbook Air M3)

| AD Mode | Function | Iterations | Original Time | Optimized Time | Speedup Factor |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Reverse AD** | Rosenbrock (10 inputs) | 100,000 | 118.36 ms | 81.55 ms | **1.45x** |
| **Reverse AD** | Polynomial (1 input) | 1,000,000 | 94.10 ms | 60.44 ms | **1.56x** |
| **Reverse AD** | Multivariate (2 inputs) | 1,000,000 | 152.50 ms | 103.57 ms | **1.47x** |
| **Forward AD** | Math Heavy (`f64x8`) | 1,000,000 | 1.12 s | 529.49 ms | **2.12x** |
| **Hessian AD** ($N=1$) | Rosenbrock (10 inputs) | 1,000 | 66.67 ms | 49.20 ms | **1.36x** |
| **Hessian AD** ($N=2$) | Rosenbrock (10 inputs) | 1,000 | 57.76 ms | 40.97 ms | **1.41x** |
| **Hessian AD** ($N=5$) | Rosenbrock (10 inputs) | 1,000 | 55.85 ms | 41.38 ms | **1.35x** |
| **Hessian AD** ($N=10$)| Rosenbrock (10 inputs) | 1,000 | 69.88 ms | 48.03 ms | **1.46x** |
| **Hessian AD** ($N=1$) | Polynomial (1 input) | 10,000 | 6.13 ms | 5.49 ms | **1.12x** |
| **Hessian AD** ($N=1$) | Multivariate (2 inputs)| 10,000 | 11.89 ms | 9.89 ms | **1.20x** |
| **Hessian AD** ($N=2$) | Multivariate (2 inputs)| 10,000 | 10.64 ms | 8.65 ms | **1.23x** |

### [0.3.0]
- **Second-Order AD**: Added full support for computing Hessians via recursive dual types.
- **New AD Modes**:
  - **Forward-over-Forward**: Using the new `HyperAD_ADFN` type.
  - **Forward-over-Reverse**: Using the new `HyperAD_ADR` type.
- **FunctionEngine Improvements**:
  - Added a high-level `.hessian()` method to `FunctionEngine` for one-call value/gradient/Hessian evaluation.
  - Implemented automatic **multi-pass batching** for Hessian computation, allowing full Hessian recovery even when the number of tangent lanes is smaller than the input dimension.
- **Enhanced Diagnostics**: Integrated `#[diagnostic::on_unimplemented]` to provide clear, actionable compiler error messages when calling Hessian methods on incompatible engines.
- **Stability**: Promoted `hessian` features from experimental to a default library feature.
- **Documentation**: Major updates to the `ad_trait` book with dedicated theory and implementation pages for second-order derivatives.

## Citation

For more information about our work, refer to our paper:
https://arxiv.org/abs/2504.15976

If you use this crate in your research, please cite:
```text
@article{liang2025ad,
  title={ad-trait: A fast and flexible automatic differentiation library in rust},
  author={Liang, Chen and Wang, Qian and Xu, Andy and Rakita, Daniel},
  journal={arXiv preprint arXiv:2504.15976},
  year={2025}
}
```
