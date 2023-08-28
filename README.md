## Introduction

This crate brings easy to use, efficient, and highly flexible automatic differentiation to the
Rust programming language.  Utilizing Rust's extensive and expressive trait
features, the several types in this crate that implement the trait AD can be thought of as a 
drop-in replacement for an f64 or f32 that affords forward mode or backwards mode automatic 
differentiation on any downstream computation in Rust.

## Key features
- ad_trait supports reverse mode or forward mode automatic differentiation.  The forward mode automatic differentiation implementation can also take advantage of SIMD to compute up to 16 lanes of tangents simultaneously.   
- The core rust f64 or f32 types also implement the AD trait, meaning any functions that take an AD trait object as a generic type can handle either standard floating point computation or derivative tracking automatic differentiation with essentially no overhead.    
- The provided types that implement the AD trait also implement several useful traits that allow it to operate almost exactly as a standard f64.  For example, it even implements the `RealField` and `ComplexField` traits,
meaning it can be used in any `nalgebra` or `ndarray` computations.
