import chex
import jax
import jax.numpy as jnp
from typing import Union
from .softmax_cross_entropy import softmax_cross_entropy

def poly_loss_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
    epsilon: float = 2.0,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    r"""Computes PolyLoss between logits and labels.

    The PolyLoss is a loss function that decomposes commonly
    used classification loss functions into a series of weighted
    polynomial bases. It is inspired by the Taylor expansion of
    cross-entropy loss and focal loss in the bases of :math:`(1 − P_t)^j`.

    Args:
      logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
      labels: Valid probability distributions (non-negative, sum to 1), e.g. a
        one hot encoding specifying the correct class for each input;
        must have a shape broadcastable to `[..., num_classes]`.
      epsilon: The coefficient of the first polynomial term.
      axis: Axis or axes along which to compute.
      where: Elements to include in the computation.

    Returns:
      Poly loss between each prediction and the corresponding target
      distributions, with shape `[...]`.
    """
    chex.assert_type([logits, labels], float)
    p = jax.nn.softmax(logits, axis=axis, where=where)
    one_minus_pt = jnp.sum(labels * (1 - p), axis=axis, where=where)
    cross_entropy = softmax_cross_entropy(
        logits=logits, labels=labels, axis=axis, where=where
    )
    return cross_entropy + epsilon * one_minus_pt
