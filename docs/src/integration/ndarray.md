# Integration with ndarray

`ad_trait` also supports the `ndarray` crate, which is commonly used for data science and machine learning.

## Example: AD Array

```rust
use ndarray::Array2;
use ad_trait::adfn;

let a = Array2::<adfn<1>>::zeros((10, 10));
```

## Scalar Operations

The `AD` trait includes `mul_by_ndarray_matrix_ref`, allowing you to perform scalar-array multiplication efficiently across different AD modes.

## Generic Algorithms

Similar to `nalgebra`, you can write generic algorithms using `ndarray` that work with any `AD` type. This is particularly useful for implementing complex mathematical models that require gradients for optimization.
