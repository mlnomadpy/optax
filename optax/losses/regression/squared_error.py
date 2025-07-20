import chex
import jax.numpy as jnp
from typing import Optional

def squared_error(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
    """Calculates the squared error for a set of predictions."""
    chex.assert_type([predictions], float)
    if targets is not None:
        chex.assert_equal_shape((predictions, targets))
    errors = predictions - targets if targets is not None else predictions
    return errors ** 2
