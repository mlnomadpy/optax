import chex
import jax.numpy as jnp

def perceptron_loss(
    predictor_outputs: chex.Numeric,
    targets: chex.Numeric
) -> chex.Numeric:
    """Binary perceptron loss.

    References:
      https://en.wikipedia.org/wiki/Perceptron

    Args:
      predictor_outputs: score produced by the model (float).
      targets: Target values. Target values should be strictly in the set {-1, 1}.

    Returns:
      loss value.
    """
    chex.assert_equal_shape([predictor_outputs, targets])
    return jnp.maximum(0, - predictor_outputs * targets)
