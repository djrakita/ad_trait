# The AD Trait

The `AD` trait is the fundamental building block of `ad_trait`. It defines the arithmetic and mathematical operations required for automatic differentiation.

## Why use a Trait?

By using a trait instead of a concrete type, `ad_trait` allows you to write generic algorithms that can be used for:
- Standard evaluation (using `f64`).
- First-order derivatives (using `adfn<1>`).
- Gradients for many inputs (using `adr`).
- Accelerated vector math (using `f64xn`).

## Key Methods

- `constant(f64) -> Self`: Creates a new AD value from a constant.
- `to_constant(&self) -> f64`: Retrieves the underlying value.
- `ad_num_mode()`: Returns the current mode (Float, ForwardAD, etc.).
- `to_other_ad_type<T2: AD>(&self) -> T2`: Converts to a different AD type.

## Numerical Operations

`AD` requires many standard numerical traits, including:
- `RealField` and `ComplexField` from `simba`.
- `num_traits::Signed`.
- Standard operator overloads (`Add`, `Mul`, etc.).

This ensures that any type implementing `AD` behaves like a sophisticated number.
