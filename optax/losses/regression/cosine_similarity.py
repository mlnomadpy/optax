import chex
import jax.numpy as jnp
from typing import Union
import functools

@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def cosine_similarity(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    """Computes the cosine similarity between targets and predictions."""
    chex.assert_type([predictions, targets], float)
    a = predictions
    b = targets
    a_norm2 = jnp.square(a).sum(axis=axis, where=where, keepdims=True)
    b_norm2 = jnp.square(b).sum(axis=axis, where=where, keepdims=True)
    a_norm = jnp.sqrt(a_norm2.clip(epsilon))
    b_norm = jnp.sqrt(b_norm2.clip(epsilon))
    a_unit = a / a_norm
    b_unit = b / b_norm
    return (a_unit * b_unit).sum(axis=axis, where=where)
