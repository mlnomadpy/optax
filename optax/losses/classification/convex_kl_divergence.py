import chex
import jax.numpy as jnp
from typing import Union

def convex_kl_divergence(
    log_predictions: chex.Array,
    log_targets: chex.Array,
    convex_combination: float = 0.5,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    """Convex combination of KL divergences between log_predictions and log_targets.

    Args:
      log_predictions: Log-probabilities of predicted distribution.
      log_targets: Log-probabilities of target distribution.
      convex_combination: Weight for the convex combination.
      axis: Axis or axes along which to compute.
      where: Elements to include in the computation.

    Returns:
      Convex combination of KL divergences.
    """
    chex.assert_type([log_predictions, log_targets], float)
    kl_pt = jnp.exp(log_targets) * (log_targets - log_predictions)
    kl_tp = jnp.exp(log_predictions) * (log_predictions - log_targets)
    loss = convex_combination * kl_pt + (1.0 - convex_combination) * kl_tp
    return jnp.sum(loss, axis=axis, where=where)
