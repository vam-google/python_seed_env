import jax.numpy as jnp
from jax import random
from jax import grad

def selu(x, alpha=1.67, lmbda=1.05):
  return lmbda * jnp.where(x > 0, x, alpha * jnp.exp(x) - alpha)

def sum_logistic(x):
  return jnp.sum(1.0 / (1.0 + jnp.exp(-x)))

x = jnp.arange(5.0)
print(selu(x))

key = random.key(1701)
x = random.normal(key, (1_000_000,))
selu(x).block_until_ready()

x_small = jnp.arange(3.)
derivative_fn = grad(sum_logistic)
print(derivative_fn(x_small))