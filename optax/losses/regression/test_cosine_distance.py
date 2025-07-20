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
"""Tests for optax.losses.regression.cosine_distance and cosine_similarity."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import numpy as np
from optax.losses.regression import cosine_distance, cosine_similarity

class CosineDistanceTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        self.ys = np.array([[10., 1., -2.], [1., 4., 0.2]], dtype=np.float32)
        self.ts = np.array([[0., 1.2, 0.2], [1., -0.3, 0.]], dtype=np.float32)
        self.exp = np.array([0.9358251989, 1.0464068465], dtype=np.float32)

    @chex.all_variants
    def test_scalar_distance(self):
        np.testing.assert_allclose(
            self.variant(cosine_distance)(self.ys[0], self.ts[0]),
            self.exp[0], atol=1e-4)

    @chex.all_variants
    def test_scalar_similarity(self):
        np.testing.assert_allclose(
            self.variant(cosine_similarity)(self.ys[0], self.ts[0]),
            1. - self.exp[0], atol=1e-4)

    @chex.all_variants
    def test_batched_distance(self):
        np.testing.assert_allclose(
            self.variant(cosine_distance)(self.ys, self.ts),
            self.exp, atol=1e-4)

    @chex.all_variants
    def test_batched_similarity(self):
        np.testing.assert_allclose(
            self.variant(cosine_similarity)(self.ys, self.ts),
            1. - self.exp, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
