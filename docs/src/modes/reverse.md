# Reverse-Mode AD

Reverse-mode automatic differentiation is the most efficient way to compute gradients of functions with a very large number of inputs (e.g., neural networks). In `ad_trait`, this is implemented using the `adr` type.

## How it Works

Unlike forward-mode, reverse-mode AD works in two phases:
1. **Forward Pass**: The function is evaluated, and all operations are recorded in a **Global Computation Graph**.
2. **Backward Pass**: The library traverses the graph in reverse, applying the chain rule to compute the gradient with respect to every input variable.

## The Global Computation Graph

Because `adr` relies on a global graph, certain care must be taken:
- The graph must be reset between independent differentiation calls (handled automatically by `FunctionEngine`).
- In multi-threaded environments, access to the graph must be synchronized.

## Usage Example

```rust
use ad_trait::ReverseAD;

// evaluate with ReverseAD
// this is best for functions with many inputs and few outputs.
let engine = FunctionEngine::new(func.clone(), func, ReverseAD::new());
```
