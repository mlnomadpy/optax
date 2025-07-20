import chex
import jax
import jax.numpy as jnp

def multiclass_hinge_loss(
    scores: chex.Array,
    labels: chex.Array,
) -> chex.Array:
    """Multiclass hinge loss.

    References:
      https://en.wikipedia.org/wiki/Hinge_loss

    Args:
      scores: scores produced by the model (floats).
      labels: ground-truth integer labels.

    Returns:
      loss values
    """
    one_hot_labels = jax.nn.one_hot(labels, scores.shape[-1])
    return (jnp.max(scores + 1.0 - one_hot_labels, axis=-1) -
            jnp.sum(scores * one_hot_labels, axis=-1))
