# Finite Differencing

Finite Differencing is a classical numerical method for estimating derivatives. It is provided in `ad_trait` as a baseline and for functions where AD types might be impractical.

## How it Works

The derivative is approximated using the formula:
$$f'(x) \approx \frac{f(x + h) - f(x)}{h}$$
where $h$ is a very small value.

## Accuracy vs. Precision

Finite differencing is an approximation and is subject to both truncation error (making $h$ too large) and round-off error (making $h$ too small). It is generally much less precise than true automatic differentiation.

## Usage Example

```rust
use ad_trait::FiniteDifferencing;

// evaluate with FiniteDifferencing
let engine = FunctionEngine::new(func.clone(), func, FiniteDifferencing::new());
```
