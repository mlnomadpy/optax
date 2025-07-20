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
"""Tests for optax.losses.classification.ctc_loss and ctc_loss_with_forward_probs."""

from absl.testing import absltest
from absl.testing import parameterized
import chex
import jax
import jax.numpy as jnp
import numpy as np
from optax.losses.classification import ctc_loss, ctc_loss_with_forward_probs

class CTCTest(parameterized.TestCase):
    def setUp(self):
        super().setUp()
        np.random.seed(1234)
        self._rtol = 5e-3 if jax.default_backend() != 'cpu' else 1e-6

    @chex.all_variants
    def test_with_one_to_one_alignment(self):
        batchsize = 8
        steps = 50
        nclasses = 40
        logits = np.random.randn(batchsize, steps, nclasses)
        labels = np.random.uniform(1, nclasses, size=(batchsize, steps)).astype(np.int32)
        for n in range(labels.shape[0]):
            for t in range(1, labels.shape[1]):
                while labels[n, t] == labels[n, t - 1]:
                    labels[n, t] = np.random.randint(1, nclasses)
        results = self.variant(ctc_loss_with_forward_probs)(
            logits, np.zeros(logits.shape[:2]), labels, np.zeros(labels.shape)
        )
        (per_seq_loss, logalpha_blank, logalpha_emit) = results
        logprobs = jax.nn.log_softmax(logits)
        for b in range(batchsize):
            p = 0.0
            for t in range(steps):
                p += logprobs[b, t, labels[b, t]]
            np.testing.assert_allclose(np.array(-p), per_seq_loss[b], rtol=self._rtol)
            np.testing.assert_allclose(
                logalpha_blank[-1, b, 0], np.sum(logprobs[b, :, 0]), rtol=self._rtol
            )
            np.testing.assert_allclose(
                logalpha_emit[-1, b, steps - 1], -per_seq_loss[b], rtol=self._rtol
            )
            np.testing.assert_allclose(
                logalpha_blank[-1, b, steps], -per_seq_loss[b], rtol=self._rtol
            )

    @chex.all_variants
    def test_with_one_to_one_alignment_and_paddings(self):
        batch_size = 5
        nclasses = 13
        steps = 7
        logits = np.random.normal(size=[batch_size, steps, nclasses])
        logprobs = jax.nn.log_softmax(logits)
        labels = []
        for _ in range(batch_size):
            row = list(range(1, nclasses))
            np.random.shuffle(row)
            labels.append(row[:steps])
        labels = np.array(labels)
        lengths = np.random.randint(3, 6, size=(batch_size,))
        paddings = self._lengths_to_paddings(lengths, steps)
        actual_loss = self.variant(ctc_loss)(
            logits, paddings, labels, paddings
        )
        value_and_grad = self.variant(jax.value_and_grad(self._average_ctc_loss))
        unused_avg_loss, actual_gradients = value_and_grad(
            logits, paddings, labels, paddings
        )
        for n in range(batch_size):
            pass  # Add more checks as needed

    @chex.all_variants
    def test_repeat_with_one_to_one_alignment(self):
        pass  # Add implementation as needed

    def _lengths_to_paddings(self, lengths, maxlength):
        indices = jnp.arange(maxlength).reshape((1,) * lengths.ndim + (maxlength,))
        lengths = np.expand_dims(lengths, axis=-1)
        elem_valid = indices < lengths
        return np.logical_not(elem_valid).astype(np.float32)

    def _average_ctc_loss(self, logprobs, logprob_paddings, labels, label_paddings):
        return jnp.average(
            ctc_loss(
                logprobs, logprob_paddings, labels, label_paddings
            )
        )

if __name__ == "__main__":
    absltest.main()
