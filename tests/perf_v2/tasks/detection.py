# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OTX object detection performance benchmark."""

from __future__ import annotations

from pathlib import Path

from tests.perf_v2.utils import (
    Criterion,
    DatasetInfo,
    ModelInfo,
)

from otx.types.task import OTXTaskType

TASK_TYPE = OTXTaskType.DETECTION

MODEL_TEST_CASES = [
    ModelInfo(task=TASK_TYPE.value, name="atss_mobilenetv2", category="default"),
    ModelInfo(task=TASK_TYPE.value, name="dfine_x", category="accuracy"),
    ModelInfo(task=TASK_TYPE.value, name="rtdetr_50", category="other"),
    ModelInfo(task=TASK_TYPE.value, name="rtmdet_tiny", category="other"),
]

DATASET_TEST_CASES = (
    [
        DatasetInfo(
            name=f"pothole_tiny_{idx}",
            path=Path("detection/pothole_small") / f"{idx}",
            group="tiny",
        )
        for idx in (1, 2, 3)
    ]
    + [
        DatasetInfo(
            name=f"wgisd_small_{idx}",
            path=Path("detection/wgisd_small") / f"{idx}",
            group="small",
        )
        for idx in (1, 2, 3)
    ]
    + [
        DatasetInfo(
            name="diopsis",
            path=Path("detection/diopsis_coco"),
            group="medium",
        ),
        DatasetInfo(
            name="pascal_tiny",
            path=Path("detection/pascal_tiny"),
            group="medium",
        ),
        DatasetInfo(
            name="bdd_large",
            path=Path("detection/bdd_large"),
            group="large",
        ),
    ]
)

BENCHMARK_CRITERIA = [
    Criterion(name="training:epoch", summary="max", compare="<", margin=0.1),
    Criterion(name="training:e2e_time", summary="max", compare="<", margin=0.1),
    Criterion(name="training:gpu_mem", summary="max", compare="<", margin=0.1),
    Criterion(name="training:train/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="training:val/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/f1-score", summary="max", compare=">", margin=0.1),
    Criterion(name="torch:test/iter_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:e2e_time", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="export:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="optimize:test/latency", summary="mean", compare="<", margin=0.1),
    Criterion(name="torch:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="export:test/e2e_time", summary="max", compare=">", margin=0.1),
    Criterion(name="optimize:test/e2e_time", summary="max", compare=">", margin=0.1),
]
