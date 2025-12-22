# Matrix Operations

`ad_trait` is designed to handle linear algebra efficiently. The `AD` trait includes requirements for matrix-scalar multiplication, which is the foundation for differentiating through linear systems.

## Scalar-Matrix Multiplication

Types implementing `AD` must provide:
- `mul_by_nalgebra_matrix`: Multiplies an AD scalar by a `nalgebra` matrix.
- `mul_by_ndarray_matrix_ref`: Multiplies an AD scalar by an `ndarray` array.

## Why this is necessary

By providing these specialized methods at the trait level, `ad_trait` can ensure that matrix operations are handled correctly for each AD mode. For example, in `ForwardAD`, the matrix multiplication also propagates the tangent through every element of the matrix.

## Performance Considerations

When performing large matrix multiplications, it is often more efficient to use a differentiation mode that minimizes the number of passes (like `ReverseAD` or `ForwardADMulti`).
