

# Block float values

This repository simulates neural network with grouping float values to same mantissa bits.

Please see the Paper(not ready) for more specific results.

Also, this repository's code can train a network with custom floating point mantissa bits.

# Files

Contents are not ready...

# Usage

Documents are not ready...

# Issues

## JAX Memory allocation issue
**Currently, we are not using JAX because functions need to be optimized better**

[JAX](https://github.com/google/jax) is Autograd and XLA, brought together for high-performance machine learning research.

This project uses jax to increase computation speed. Since this project uses jax with pytorch, using jax's basic config of memory allocation has high possibility to run out of memory.

Type

```
export XLA_PYTHON_CLIENT_MEM_FRACTION=.30
```

or 

```
export XLA_PYTHON_CLIENT_PREALLOCATE=false
```

to avoid allocating memory to jax too much.

# Requirements
torch >= 1.7.1

torchvision >= 0.5.0
