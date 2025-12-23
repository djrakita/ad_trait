# ad_trait

[![](https://img.shields.io/crates/v/ad_trait.svg)](https://crates.io/crates/ad_trait) [![](https://img.shields.io/badge/docs-book-blue)](https://djrakita.github.io/ad_trait/) [![](https://docs.rs/ad_trait/badge.svg)](https://docs.rs/ad_trait) [![](https://github.com/djrakita/ad_trait/actions/workflows/deploy_book.yml/badge.svg)](https://github.com/djrakita/ad_trait/actions/workflows/deploy_book.yml)

# Introduction
This crate brings easy to use, efficient, and highly flexi
Rust programming language. Utilizing Rust's extensive and
types in this crate that implement the trait AD can be tho
f64 or f32 that affords forward mode or backwards mode aut
computation in Rust.
# Key Features
- ad_trait supports reverse mode or forward mode automatic differentiation. The forward mode automatic
differentiation implementation can also take advantage of SIMD to compute multiple tangents simultaneously.
- The core rust f64 or f32 types also implement the AD trait, meaning any functions that take an AD
trait object as a generic type can handle either standard floating point computation or derivative
tracking automatic differentiation with essentially no overhead.
- The provided types that implement the AD trait also implement several useful traits that allow it
to operate almost exactly as a standard f64. For example, it even implements the `RealField` and
`ComplexField` traits, meaning it can be used in any `nalgebra` or `ndarray` computations.

# Example
```rust
use ad_trait::AD;
use ad_trait::differentiable_block::DifferentiableBlock;
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
  let differentiable_block = DifferentiableBlock::new(function_standard, function_derivative, ReverseAD::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Reverse AD: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Forward AD //////////////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.to_other_ad_type::<adfn<1>>();
  let differentiable_block = DifferentiableBlock::new(function_standard, function_derivative, ForwardAD::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Forward AD: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Forward AD Multi ////////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.to_other_ad_type::<adfn<2>>();
  let differentiable_block = DifferentiableBlock::new(function_standard, function_derivative, ForwardADMulti::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Forward AD Multi: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

  // Finite Differencing /////////////////////////////////////////////////////////////////////////
  let function_standard = Test { coeff: 2.0 };
  let function_derivative = function_standard.clone();
  let differentiable_block = DifferentiableBlock::new(function_standard, function_derivative, FiniteDifferencing::new());

  let (f_res, derivative_res) = differentiable_block.derivative(&inputs);
  println!("Finite Differencing: ");
  println!("  f_res: {}", f_res[0]);
  println!("  derivative: {}", derivative_res);
  println!("//////////////");
  println!();

}
```

# Citation

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
