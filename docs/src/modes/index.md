# Automatic Differentiation Modes

`ad_trait` supports three primary modes of differentiation, each with its own strengths and weaknesses.

## Summary Table

| Mode | Type | Speed (Few Inputs) | Speed (Many Inputs) | Precision |
| :--- | :--- | :--- | :--- | :--- |
| **Forward-Mode** | Exact | Very Fast | Slow | Exact |
| **Reverse-Mode** | Exact | Moderate | Very Fast | Exact |
| **Finite Differencing** | Approx | Fast | Very Slow | Low |

In the following chapters, we will explore each of these modes in detail.
