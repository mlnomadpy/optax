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
"""Tests for optax.losses.classification.kl_divergence and related functions."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses.classification import kl_divergence, kl_divergence_with_log_targets, convex_kl_divergence

class KLDivergenceTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.log_ps = np.array([
            [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
            [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
        ])
        self.qs = np.array(
            [[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], [0.05, 0.03, 0.02, 0.3, 0.5, 0.0]]
        )
        self.exp = np.array([0.8875577, 0.7592807])

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(kl_divergence)(self.log_ps[0], self.qs[0]),
            self.exp[0],
            atol=1e-4,
        )

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(kl_divergence)(self.log_ps, self.qs),
            self.exp,
            atol=1e-4,
        )

    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask(self, size):
        preds = np.random.normal(size=size)
        targets = np.random.dirichlet(np.ones(size))
        mask = np.random.randint(2, size=size, dtype=bool)
        x = kl_divergence(preds[mask], targets[mask])
        y = kl_divergence(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
        x = kl_divergence(preds, targets, axis=axis)
        y = kl_divergence(
            np.moveaxis(preds, axis, -1),
            np.moveaxis(targets, axis, -1),
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

class KLDivergenceWithLogTargetsTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.log_ps = np.array([
            [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
            [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
        ])
        self.qs = np.array([
            [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
            [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
        ])
        self.exp = np.array([0.8875625, 0.7187435584901326])

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(kl_divergence_with_log_targets)(self.log_ps[0], self.qs[0]),
            self.exp[0],
            atol=1e-4,
        )

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(kl_divergence_with_log_targets)(self.log_ps, self.qs),
            self.exp,
            atol=1e-4,
        )

    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask(self, size):
        preds = np.random.normal(size=size)
        targets = np.log(np.random.dirichlet(np.ones(size)))
        mask = np.random.randint(2, size=size, dtype=bool)
        f = kl_divergence_with_log_targets
        x = f(preds[mask], targets[mask])
        y = f(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.log(np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1]))
        f = kl_divergence_with_log_targets
        x = f(preds, targets, axis=axis)
        y = f(
            np.moveaxis(preds, axis, -1),
            np.moveaxis(targets, axis, -1),
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

class ConvexKLDivergenceTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.log_ps = np.array([
            [-2.9957, -3.5066, -3.9120, -1.2040, -0.6931, -2.3026],
            [-1.6094, -1.6094, -1.6094, -2.3026, -1.8971, -1.8971],
        ])
        self.qs = np.array(
            [[0.2, 0.2, 0.2, 0.1, 0.15, 0.15], [0.05, 0.03, 0.02, 0.3, 0.5, 0.0]]
        )
        self.exp = np.array([0.88757247, 0.859308])

    @chex.all_variants
    def test_scalar(self):
        np.testing.assert_allclose(
            self.variant(convex_kl_divergence)(self.log_ps[0], self.qs[0]),
            self.exp[0],
            atol=1e-4,
        )

    @chex.all_variants
    def test_batched(self):
        np.testing.assert_allclose(
            self.variant(convex_kl_divergence)(self.log_ps, self.qs),
            self.exp,
            atol=1e-4,
        )

    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask(self, size):
        preds = np.random.normal(size=size)
        targets = np.random.dirichlet(np.ones(size))
        mask = np.random.randint(2, size=size, dtype=bool)
        x = convex_kl_divergence(preds[mask], targets[mask])
        y = convex_kl_divergence(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.random.dirichlet(np.ones(shape[-1]), size=shape[:-1])
        x = convex_kl_divergence(preds, targets, axis=axis)
        y = convex_kl_divergence(
            np.moveaxis(preds, axis, -1),
            np.moveaxis(targets, axis, -1),
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
