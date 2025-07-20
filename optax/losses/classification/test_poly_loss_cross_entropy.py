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
"""Tests for optax.losses.classification.poly_loss_cross_entropy."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from optax.losses.classification import poly_loss_cross_entropy, softmax_cross_entropy

class PolyLossTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.logits = np.array([0.14, 1.456, 2.356, -0.124, -2.47])
        self.labels = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        self.batched_logits = np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]])
        self.batched_labels = np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]])

    @chex.all_variants
    @parameterized.parameters(
        dict(eps=2, expected=4.5317),
        dict(eps=1, expected=3.7153),
        dict(eps=-1, expected=2.0827),
        dict(eps=0, expected=2.8990),
        dict(eps=-0.5, expected=2.4908),
        dict(eps=1.15, expected=3.8378),
        dict(eps=1.214, expected=3.8900),
        dict(eps=5.45, expected=7.3480),
    )
    def test_scalar(self, eps, expected):
        np.testing.assert_allclose(
            self.variant(poly_loss_cross_entropy)(self.logits, self.labels, epsilon=eps),
            expected,
            atol=1e-4,
        )

    @chex.all_variants
    @parameterized.parameters(
        dict(eps=2, expected=np.array([0.4823, 1.2567])),
        dict(eps=1, expected=np.array([0.3261, 1.0407])),
        dict(eps=0, expected=np.array([0.1698, 0.8247])),
        dict(eps=-0.5, expected=np.array([0.0917, 0.7168])),
        dict(eps=1.15, expected=np.array([0.3495, 1.0731])),
        dict(eps=1.214, expected=np.array([0.3595, 1.0870])),
        dict(eps=5.45, expected=np.array([1.0211, 2.0018])),
    )
    def test_batched(self, eps, expected):
        np.testing.assert_allclose(
            self.variant(poly_loss_cross_entropy)(self.batched_logits, self.batched_labels, epsilon=eps),
            expected,
            atol=1e-4,
        )

    @chex.all_variants
    @parameterized.parameters(
        dict(
            logits=np.array(
                [[4.0, 2.0, 1.0], [0.0, 5.0, 1.0], [0.134, 1.234, 3.235]]
            ),
            labels=np.array(
                [[1.0, 0.0, 0.0], [0.0, 0.8, 0.2], [0.34, 0.33, 0.33]]
            ),
        ),
        dict(
            logits=np.array([[4.0, 2.0, 1.0], [0.0, 5.0, 1.0]]),
            labels=np.array([[1.0, 0.0, 0.0], [0.0, 0.8, 0.2]]),
        ),
        dict(
            logits=np.array(
                [[4.0, 2.0, 1.0, 0.134, 1.3515], [0.0, 5.0, 1.0, 0.5215, 5.616]]
            ),
            labels=np.array(
                [[0.5, 0.0, 0.0, 0.0, 0.5], [0.0, 0.12, 0.2, 0.56, 0.12]]
            ),
        ),
        dict(logits=np.array([1.89, 2.39]), labels=np.array([0.34, 0.66])),
        dict(logits=np.array([0.314]), labels=np.array([1.0])),
    )
    def test_equals_to_cross_entropy_when_eps0(self, logits, labels):
        np.testing.assert_allclose(
            self.variant(poly_loss_cross_entropy)(logits, labels, epsilon=0.0),
            self.variant(softmax_cross_entropy)(logits, labels),
            atol=1e-4,
        )

    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask(self, size):
        preds = np.random.normal(size=size)
        targets = np.random.dirichlet(np.ones(size))
        mask = np.random.randint(2, size=size, dtype=bool)
        x = poly_loss_cross_entropy(preds[mask], targets[mask])
        y = poly_loss_cross_entropy(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
        x = poly_loss_cross_entropy(preds, targets, axis=axis)
        y = poly_loss_cross_entropy(
            np.moveaxis(preds, axis, -1),
            np.moveaxis(targets, axis, -1),
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
