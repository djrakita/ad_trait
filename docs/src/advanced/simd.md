# SIMD Acceleration

`ad_trait` provides first-class support for Single Instruction, Multiple Data (SIMD) acceleration through the `f64xn<N>` type.

## What is SIMD?

SIMD allows a single CPU instruction to operate on multiple pieces of data (usually vectors) at once. This can lead to 4x-8x speedups for arithmetic operations on modern processors.

## Using `f64xn`

The `f64xn<N>` type implements the `AD` trait (in `SIMDNum` mode). It stores $N$ floats and performs element-wise operations using SIMD intrinsics (where available).

## Requirements

Currently, `f64xn` requires the **nightly** version of Rust to access the `portable_simd` feature.

## Example

```rust
// Requires nightly and #![feature(portable_simd)]
use ad_trait::simd::f64xn;

let a = f64xn::<4>::new([1.0, 2.0, 3.0, 4.0]);
let b = f64xn::<4>::new([5.0, 6.0, 7.0, 8.0]);
let c = a + b; // Computed using a single CPU instruction if possible
```
