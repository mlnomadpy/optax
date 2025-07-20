import chex
import jax.numpy as jnp
from typing import Union

def kl_divergence_with_log_targets(
    log_predictions: chex.Array,
    log_targets: chex.Array,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    """KL divergence where both predictions and targets are in log-space.

    Args:
      log_predictions: Log-probabilities of predicted distribution.
      log_targets: Log-probabilities of target distribution.
      axis: Axis or axes along which to compute.
      where: Elements to include in the computation.

    Returns:
      KL divergence between predicted and target distributions.
    """
    chex.assert_type([log_predictions, log_targets], float)
    loss = jnp.exp(log_targets) * (log_targets - log_predictions)
    return jnp.sum(loss, axis=axis, where=where)
