# Copyright 2019 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Tests for optax.losses.classification.sparsemax_loss and multiclass_sparsemax_loss."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses.classification import sparsemax_loss, multiclass_sparsemax_loss

class SparsemaxTest(parameterized.TestCase):
    def test_binary(self):
        label = 1
        score = 10.0
        def reference_impl(label, logit):
            scores = -(2 * label - 1) * logit
            if scores <= -1.0:
                return 0.0
            elif scores >= 1.0:
                return scores
            else:
                return (scores + 1.0) ** 2 / 4
        expected = reference_impl(label, score)
        result = sparsemax_loss(jnp.asarray(score), jnp.asarray(label))
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_batched_binary(self):
        labels = jnp.array([1, 0])
        scores = jnp.array([10.0, 20.0])
        def reference_impl(label, logit):
            scores = -(2 * label - 1) * logit
            if scores <= -1.0:
                return 0.0
            elif scores >= 1.0:
                return scores
            else:
                return (scores + 1.0) ** 2 / 4
        expected = jnp.asarray([
            reference_impl(labels[0], scores[0]),
            reference_impl(labels[1], scores[1]),
        ])
        result = sparsemax_loss(scores, labels)
        np.testing.assert_allclose(result, expected, atol=1e-4)

    def test_multi_class_zero_loss(self):
        labels = jnp.array([1, 0, 2])
        scores = jnp.array([[0.0, 1e5, 0.0],
                            [1e5, 0.0, 0.0],
                            [0.0, 0.0, 1e5]])
        losses = multiclass_sparsemax_loss(scores, labels)
        np.testing.assert_allclose(losses, np.array([0.0, 0.0, 0.0]), atol=1e-4)

    def test_multi_class_gradient(self):
        def loss_mean(scores, labels):
            return jnp.mean(multiclass_sparsemax_loss(scores, labels))
        labels = jnp.array([1, 0, 2])
        scores = jnp.array([[0.0, 1e5, 0.0],
                            [1e5, 0.0, 0.0],
                            [0.0, 0.0, 1e5]])
        grad = jax.grad(loss_mean)(scores, labels)
        projection_vmap = jax.vmap(lambda x: x)  # Placeholder for projections.projection_simplex
        grad_expected = projection_vmap(scores) - jax.nn.one_hot(labels, 3)
        np.testing.assert_allclose(grad, grad_expected, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
