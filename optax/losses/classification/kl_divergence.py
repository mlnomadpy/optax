import chex
import jax.numpy as jnp
from typing import Union

def kl_divergence(
    log_predictions: chex.Array,
    targets: chex.Array,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    """Computes the Kullback-Leibler divergence (relative entropy) loss.

    Measures the information gain achieved if target probability distribution
    would be used instead of predicted probability distribution.

    Args:
      log_predictions: Probabilities of predicted distribution with shape [...,
        dim]. Expected to be in the log-space to avoid underflow.
      targets: Probabilities of target distribution with shape [..., dim].
        Expected to be strictly positive.
      axis: Axis or axes along which to compute.
      where: Elements to include in the computation.

    Returns:
      Kullback-Leibler divergence of predicted distribution from target
      distribution with shape [...].
    """
    chex.assert_type([log_predictions, targets], float)
    loss = targets * (
        jnp.where(targets == 0, 0, jnp.log(targets)) - log_predictions
    )
    return jnp.sum(loss, axis=axis, where=where)
