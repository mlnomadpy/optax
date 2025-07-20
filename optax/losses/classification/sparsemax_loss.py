import chex
import jax
import jax.numpy as jnp

def sparsemax_loss(
    logits: chex.Array,
    labels: chex.Array,
) -> chex.Array:
    """Binary sparsemax loss.

    This loss is zero if and only if `jax.nn.sparse_sigmoid(logits) == labels`.

    References:
      Learning with Fenchel-Young Losses. Mathieu Blondel, André F. T. Martins,
      Vlad Niculae. JMLR 2020. (Sec. 4.4)

    Args:
      logits: score produced by the model (float).
      labels: ground-truth integer label (0 or 1).

    Returns:
      loss value

    .. versionadded:: 0.2.3
    """
    return jax.nn.sparse_plus(jnp.where(labels, -logits, logits))
