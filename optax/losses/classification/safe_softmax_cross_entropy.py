import chex
import jax.numpy as jnp
from .weighted_logsoftmax import weighted_logsoftmax

def safe_softmax_cross_entropy(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array:
    """Computes the softmax cross entropy between sets of logits and labels.

    Contrarily to :func:`optax.softmax_cross_entropy` this function handles
    ``labels*logsoftmax(logits)`` as ``0`` when ``logits=-inf`` and ``labels=0``,
    following the convention that ``0 log 0 = 0``.

    Args:
      logits: Unnormalized log probabilities, with shape `[..., num_classes]`.
      labels: Valid probability distributions (non-negative, sum to 1), e.g a
        one hot encoding specifying the correct class for each input;
        must have a shape broadcastable to `[..., num_classes]`.

    Returns:
      cross entropy between each prediction and the corresponding target
      distributions, with shape `[...]`.
    """
    chex.assert_type([logits], float)
    return -jnp.sum(weighted_logsoftmax(logits, labels), axis=-1)
