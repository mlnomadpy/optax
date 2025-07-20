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
"""The losses sub-package."""

# pylint:disable=g-importing-member

from optax.losses.classification import (
    sigmoid_binary_cross_entropy,
    binary_logistic_loss,
    hinge_loss,
    perceptron_loss,
    sparsemax_loss,
    binary_sparsemax_loss,
    weighted_logsoftmax,
    safe_softmax_cross_entropy,
    softmax_cross_entropy,
    softmax_cross_entropy_with_integer_labels,
    multiclass_logistic_loss,
    multiclass_hinge_loss,
    multiclass_perceptron_loss,
    poly_loss_cross_entropy,
    kl_divergence,
    kl_divergence_with_log_targets,
    convex_kl_divergence,
    ctc_loss_with_forward_probs,
    ctc_loss,
    sigmoid_focal_loss,
    multiclass_sparsemax_loss,
)
from optax.losses._fenchel_young import make_fenchel_young_loss
from optax.losses._ranking import ranking_softmax_loss
from optax.losses._regression import cosine_distance
from optax.losses._regression import cosine_similarity
from optax.losses._regression import huber_loss
from optax.losses._regression import l2_loss
from optax.losses._regression import log_cosh
from optax.losses._regression import squared_error
from optax.losses._self_supervised import ntxent
from optax.losses._smoothing import smooth_labels
