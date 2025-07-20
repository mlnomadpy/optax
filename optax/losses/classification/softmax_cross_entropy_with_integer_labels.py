import chex
import jax
import jax.numpy as jnp
from typing import Union

def softmax_cross_entropy_with_integer_labels(
    logits: chex.Array,
    labels: chex.Array,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    r"""Computes softmax cross entropy between the logits and integer labels.

    This loss is useful for classification problems with integer labels that are
    not one-hot encoded. This loss is also known as categorical cross entropy.

    Args:
      logits: Unnormalized log probabilities, with shape ``[batch_size,
        num_classes]``.
      labels: Integers specifying the correct class for each input, with shape
        ``[batch_size]``. Class labels are assumed to be between 0 and
        ``num_classes - 1`` inclusive.
      axis: Axis or axes along which to compute.
      where: Elements to include in the computation.

    Returns:
      Cross-entropy between each prediction and the corresponding target
      distributions, with shape ``[batch_size]``.
    """
    chex.assert_type([logits], float)
    chex.assert_type([labels], int)
    logits_max = jnp.max(
        logits, axis, keepdims=True, where=where, initial=-jnp.inf
    )
    logits -= jax.lax.stop_gradient(logits_max)
    label_logits = jnp.take_along_axis(
        logits, jnp.expand_dims(labels, axis), axis=axis
    ).take(0, axis=axis)
    log_normalizers = jnp.log(jnp.sum(jnp.exp(logits), axis=axis, where=where))
    return log_normalizers - label_logits
