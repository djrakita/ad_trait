# Introduction

`ad_trait` is a powerful, flexible, and easy-to-use Automatic Differentiation (AD) library for Rust. 

## What is Automatic Differentiation?

Automatic Differentiation is a set of techniques to numerically evaluate the derivative of a function specified by a computer program. Unlike symbolic differentiation, which produces a mathematical expression for the derivative, or finite differencing, which estimates derivatives from function evaluations, AD computes derivatives exactly (up to machine precision) by applying the chain rule to the program's elementary operations.

## Why ad_trait?

- **Unified Interface**: Use the same code for forward-mode, reverse-mode, and finite differencing.
- **Integration**: Seamlessly works with `nalgebra` and `ndarray`.
- **Flexibility**: Define your own differentiable functions using a simple trait.
- **Performance**: High-performance implementations including multi-tangent forward AD and SIMD acceleration.

## Goals

The primary goal of `ad_trait` is to make sophisticated automatic differentiation accessible to the Rust ecosystem with a focus on robotics, optimization, and machine learning.
