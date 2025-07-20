import chex
import jax.numpy as jnp
from typing import Optional
from .squared_error import squared_error

import functools

@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def huber_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
    delta: float = 1.
) -> chex.Array:
    """Huber loss, similar to L2 loss close to zero, L1 loss away from zero."""
    chex.assert_type([predictions], float)
    errors = (predictions - targets) if (targets is not None) else predictions
    abs_errors = jnp.abs(errors)
    quadratic = jnp.minimum(abs_errors, delta)
    linear = abs_errors - quadratic
    return 0.5 * quadratic ** 2 + delta * linear
