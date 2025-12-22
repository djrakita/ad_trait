# Forward-Mode AD

Forward-mode automatic differentiation propagates derivatives along with the function evaluation. In `ad_trait`, this is implemented using the `adfn<N>` type.

## How it Works

A forward-mode AD variable can be thought of as a pair $(v, \dot{v})$, where $v$ is the current value and $\dot{v}$ is its tangent (the derivative with respect to some input). Every operation on these variables updates both the value and the tangent using the rules of calculus.

## Single-Tangent Forward AD

When using `ForwardAD`, the library calculates the derivative with respect to one input at a time. To compute a full Jacobian for $M$ inputs, the function is evaluated $M$ times.

## Multi-Tangent Forward AD

One of the unique features of `ad_trait` is its support for multiple tangents. By using `adfn<N>`, you can compute up to $N$ columns of the Jacobian in a single forward pass. This is extremely efficient for functions where most work is shared across different input variables.

## Usage Example

```rust
use ad_trait::{ForwardAD, adfn};

// evaluate with ForwardAD
// this will compute derivatives by calling the function once 
// per input dimension.
let engine = FunctionEngine::new(func.clone(), func, ForwardAD::new());
```
