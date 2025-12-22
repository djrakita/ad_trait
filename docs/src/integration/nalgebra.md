# Integration with nalgebra

`ad_trait` types are fully compatible with `nalgebra`'s generic matrix types.

## Example: AD Matrix

You can create a `nalgebra` matrix using any `AD` type:

```rust
use nalgebra::SMatrix;
use ad_trait::adfn;

// A 2x2 matrix of Forward-Mode AD variables
let m = SMatrix::<adfn<1>, 2, 2>::zeros();
```

## Differentiating through nalgebra

Because `adfn` and `adr` implement the relevant traits required by `nalgebra` (like `RealField` and `ComplexField`), you can use standard `nalgebra` functions (determinant, inverse, multiplication) in your differentiable code.

```rust
fn my_func<T: AD>(inputs: &[T]) -> Vec<T> {
    let m = SMatrix::<T, 2, 2>::from_vec(inputs.to_vec());
    let inv = m.try_inverse().unwrap();
    vec![inv[(0, 0)]]
}
```
