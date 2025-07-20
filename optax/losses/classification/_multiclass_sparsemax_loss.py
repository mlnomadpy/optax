import chex
import jax
import jax.numpy as jnp
from optax import projections

def _multiclass_sparsemax_loss(
    scores: chex.Array, label: chex.Array
) -> chex.Array:
    scores = jnp.asarray(scores)
    proba = projections.projection_simplex(scores)
    scores = (scores - scores[label]).at[label].set(0.0)
    return (jnp.dot(proba, jnp.where(proba, scores, 0.0))
            + 0.5 * (1.0 - jnp.dot(proba, proba)))
