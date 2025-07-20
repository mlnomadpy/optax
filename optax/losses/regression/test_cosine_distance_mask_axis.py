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
"""Tests for optax.losses.regression.cosine_distance and cosine_similarity mask and axis options."""

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
from optax.losses.regression import cosine_distance, cosine_similarity

class CosineDistanceMaskAxisTest(parameterized.TestCase):
    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask_distance(self, size):
        preds = np.random.normal(size=size)
        targets = np.random.normal(size=size)
        mask = np.random.randint(2, size=size, dtype=bool)
        x = cosine_distance(preds[mask], targets[mask])
        y = cosine_distance(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(dict(size=5), dict(size=10))
    def test_mask_similarity(self, size):
        preds = np.random.normal(size=size)
        targets = np.random.normal(size=size)
        mask = np.random.randint(2, size=size, dtype=bool)
        x = cosine_similarity(preds[mask], targets[mask])
        y = cosine_similarity(preds, targets, where=mask)
        np.testing.assert_allclose(x, y, atol=1e-4)

    @parameterized.parameters(
        dict(axis=0, shape=[4, 5, 6]),
        dict(axis=1, shape=[4, 5, 6]),
        dict(axis=2, shape=[4, 5, 6]),
    )
    def test_axis(self, shape, axis):
        preds = np.random.normal(size=shape)
        targets = np.random.normal(size=shape)
        x = cosine_similarity(preds, targets, axis=axis)
        y = cosine_similarity(
            np.moveaxis(preds, axis, -1),
            np.moveaxis(targets, axis, -1),
        )
        np.testing.assert_allclose(x, y, atol=1e-4)

if __name__ == "__main__":
    absltest.main()
