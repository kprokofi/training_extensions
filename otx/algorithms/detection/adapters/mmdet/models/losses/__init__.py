"""Loss list of mmdetection adapters."""
# Copyright (C) 2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from .cross_focal_loss import CrossSigmoidFocalLoss
from .l2sp_loss import L2SPLoss
from .cross_entropy_bpm_loss import CrossEntropyLossBPM

__all__ = ["CrossSigmoidFocalLoss", "L2SPLoss", "CrossEntropyLossBPM"]
