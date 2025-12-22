# Core Concepts

`ad_trait` is built around a few central abstractions that make it highly extensible.

## The AD Type System

The cornerstone of the library is the `AD` trait. Any type that implements `AD` can be used in a differentiable computation. The library provides several built-in implementations:

- `f64` and `f32`: For standard computations without derivative tracking.
- `adfn<N>`: For forward-mode AD with $N$ tangents.
- `adr`: For reverse-mode AD using a global computation graph.
- `f64xn<N>`: For SIMD-accelerated numerical computations.

## Trait Hierarchy

1. **`AD`**: The base numerical trait for differentiation.
2. **`DifferentiableFunctionTrait<T>`**: Defines how a function is evaluated for a given AD type `T`.
3. **`Reparameterize`**: Bridges the gap between different AD types, allowing a function to be automatically adapted for different differentiation modes.
4. **`DerivativeMethodTrait`**: Defines how a derivative is calculated (e.g., Forward, Reverse).

## The Function Engine

The `FunctionEngine` is the primary interface for users. It wraps a differentiable function and a derivative method, providing a simple way to call the function and get its Jacobian.
