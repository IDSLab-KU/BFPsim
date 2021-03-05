

# Block float values


## JAX Memory allocation issue
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

## Requirements
torch >= 1.7.1
torchvision >= 0.5.0
jax >= 0.2.9