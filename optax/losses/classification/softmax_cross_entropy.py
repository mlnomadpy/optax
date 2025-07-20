import chex
import jax
import jax.numpy as jnp
from typing import Union

def softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    r"""Computes the softmax cross entropy between sets of logits and labels.

    This loss function is commonly used for multi-class classification tasks. It
    measures the dissimilarity between the predicted probability distribution
    (obtained by applying the softmax function to the logits) and the true
    probability distribution (represented by the one-hot encoded labels).
    This loss is also known as categorical cross entropy.

    Args:
      logits: Unnormalized log probabilities, with shape ``[batch_size,
        num_classes]``.
      labels: One-hot encoded labels, with shape `[batch_size, num_classes]`. Each
        row represents the true class distribution for a single example.
      axis: Axis or axes along which to compute.
      where: Elements to include in the computation.

    Returns:
      Cross-entropy between each prediction and the corresponding target
      distributions, with shape ``[batch_size]``.
    """
    chex.assert_type([logits], float)
    log_probs = jax.nn.log_softmax(logits, axis, where)
    return -(labels * log_probs).sum(axis, where=where)
