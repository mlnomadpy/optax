import chex
from typing import Optional
from .squared_error import squared_error

def l2_loss(
    predictions: chex.Array,
    targets: Optional[chex.Array] = None,
) -> chex.Array:
    """Calculates the L2 loss for a set of predictions."""
    return 0.5 * squared_error(predictions, targets)
