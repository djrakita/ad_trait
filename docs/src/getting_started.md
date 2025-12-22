# Getting Started

Adding `ad_trait` to your project is straightforward.

## Installation

Add the following to your `Cargo.toml`:

```toml
[dependencies]
ad_trait = "0.2.0"
```

## Basic Usage

The core workflow of `ad_trait` involves three steps:

1. **Implement `DifferentiableFunctionTrait`**: Define your function.
2. **Implement `Reparameterize`**: Allow your function to work with different AD types.
3. **Use `FunctionEngine`**: Wrap your function with a differentiation method.

### A Simple Example

Here's how to compute the derivative of $f(x) = x^2$:

```rust
use ad_trait::{AD, DifferentiableFunctionTrait, Reparameterize, FunctionEngine, ForwardAD};

#[derive(Clone)]
struct Square;

impl<T: AD> DifferentiableFunctionTrait<T> for Square {
    const NAME: &'static str = "Square";
    fn call(&self, inputs: &[T], _freeze: bool) -> Vec<T> {
        vec![inputs[0] * inputs[0]]
    }
    fn num_inputs(&self) -> usize { 1 }
    fn num_outputs(&self) -> usize { 1 }
}

impl Reparameterize for Square {
    type SelfType<T2: AD> = Square;
}

fn main() {
    let func = Square;
    let engine = FunctionEngine::new(func.clone(), func, ForwardAD::new());
    
    let x = 3.0;
    let (val, grad) = engine.derivative(&[x]);
    
    println!("f(3) = {}", val[0]); // Output: 9
    println!("f'(3) = {}", grad[(0, 0)]); // Output: 6
}
```
