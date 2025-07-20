import chex
import jax
import jax.numpy as jnp

def multiclass_perceptron_loss(
    scores: chex.Array,
    labels: chex.Array,
) -> chex.Array:
    """Multiclass perceptron loss.

    References:
      Michael Collins. Discriminative training methods for Hidden Markov Models:
      Theory and experiments with perceptron algorithms. EMNLP 2002

    Args:
      scores: scores produced by the model.
      labels: ground-truth integer labels.

    Returns:
      loss values.
    """
    one_hot_labels = jax.nn.one_hot(labels, scores.shape[-1])
    return jnp.max(scores, axis=-1) - jnp.sum(scores * one_hot_labels, axis=-1)
