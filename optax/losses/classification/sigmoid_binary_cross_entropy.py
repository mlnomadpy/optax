import chex
import jax
import jax.numpy as jnp

def sigmoid_binary_cross_entropy(logits, labels):
    """Computes element-wise sigmoid cross entropy given logits and labels.

    This function can be used for binary or multiclass classification (where each
    class is an independent binary prediction and different classes are not
    mutually exclusive e.g. predicting that an image contains both a cat
    and a dog.)

    Because this function is overloaded, please ensure your `logits` and `labels`
    are compatible with each other. If you're passing in binary `labels` (values
    in {0, 1}), ensure your `logits` correspond to class 1 only. If you're
    passing in per-class target probabilities or one-hot `labels`, please ensure
    your `logits` are also multiclass. Be particularly careful if you're relying
    on implicit broadcasting to reshape `logits` or `labels`.

    References:
      [Goodfellow et al, 2016](http://www.deeplearningbook.org/contents/prob.html)

    Args:
      logits: Each element is the unnormalized log probability of a binary
        prediction. See note about compatibility with `labels` above.
      labels: Binary labels whose values are {0,1} or multi-class target
        probabilities. See note about compatibility with `logits` above.

    Returns:
      cross entropy for each binary prediction, same shape as `logits`.
    """
    chex.assert_type([logits], float)
    labels = labels.astype(logits.dtype)
    log_p = jax.nn.log_sigmoid(logits)
    log_not_p = jax.nn.log_sigmoid(-logits)
    return -labels * log_p - (1. - labels) * log_not_p
