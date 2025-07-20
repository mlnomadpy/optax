import chex
import jax.numpy as jnp

def hinge_loss(
    predictor_outputs: chex.Array,
    targets: chex.Array
) -> chex.Array:
    """Computes the hinge loss for binary classification.

    Args:
      predictor_outputs: Outputs of the decision function.
      targets: Target values. Target values should be strictly in the set {-1, 1}.

    Returns:
      loss value.
    """
    return jnp.maximum(0, 1 - predictor_outputs * targets)
