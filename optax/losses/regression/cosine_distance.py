import chex
from typing import Union
import functools
from .cosine_similarity import cosine_similarity

@functools.partial(chex.warn_only_n_pos_args_in_future, n=2)
def cosine_distance(
    predictions: chex.Array,
    targets: chex.Array,
    epsilon: float = 0.,
    axis: Union[int, tuple[int, ...], None] = -1,
    where: Union[chex.Array, None] = None,
) -> chex.Array:
    """Computes the cosine distance between targets and predictions."""
    chex.assert_type([predictions, targets], float)
    return 1.0 - cosine_similarity(
        predictions, targets, epsilon=epsilon, axis=axis, where=where
    )
