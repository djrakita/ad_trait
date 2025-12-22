# Multi-Tangents

Multi-tangent forward AD is a powerful optimization technique that allows `ad_trait` to compute multiple partial derivatives in a single pass of the function.

## The Problem

Standard forward AD computes one column of the Jacobian per pass. If a function has 100 inputs, you need 100 passes. For complex functions, this overhead can be significant.

## The Solution: `adfn<N>`

By setting $N > 1$, the tangent component of the AD type becomes a vector of length $N$. Each multiplication or operation now operates on this vector simultaneously.

## When to Use

- When the input dimension is moderately large (e.g., 2-32).
- When the function evaluation is computationally expensive relative to the number of inputs.
- When you want to minimize the number of times the function logic is executed.

## Integration with SIMD

Multi-tangents pair perfectly with SIMD. By propagating multiple tangents, the underlying hardware can often execute these operations in parallel, providing a "free" speedup.
