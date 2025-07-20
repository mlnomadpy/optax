import chex
import jax.numpy as jnp
from typing import Optional

def log_cosh(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
    """Calculates the log-cosh loss for a set of predictions."""
    chex.assert_type([predictions], float)
    errors = (predictions - targets) if (targets is not None) else predictions
    return jnp.logaddexp(errors, -errors) - jnp.log(2.0).astype(errors.dtype)
