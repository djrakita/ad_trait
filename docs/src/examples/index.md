# Examples

The `ad_trait` repository contains several examples that demonstrate the library in action.

## Built-in Examples

Check the `examples/` directory in the crate for standalone programs:
- `test.rs`: A general demonstration of forward and reverse AD.

## Regression Tests

The `tests/regression_tests.rs` file contains many examples of differentiating complex functions, including:
- Multi-variate polynomials.
- Matrix-vector multiplication.
- Jacobian calculations for multiple outputs.

These tests serve as an excellent reference for how to structure your own differentiable functions.
