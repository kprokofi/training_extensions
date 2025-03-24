DATA_ROOT=./xpu_benchmark_datasets
OUTPUT_ROOT=./benchmark_otx_2.4.0_xpu
DATA_GROUP=all
EVAL_UPTO=optimize
NUM_REPEAT=3
DEVICE=xpu

# cls
pytest tests/perf/test_classification.py\
    --model-category all \
    --data-group $DATA_GROUP \
    --eval-upto $EVAL_UPTO \
    --data-root $DATA_ROOT \
    --output-root $OUTPUT_ROOT \
    --device $DEVICE \
    --num-repeat $NUM_REPEAT \
    --verbose

# det
pytest tests/perf/test_detection.py \
    --model-category all \
    --data-group $DATA_GROUP \
    --eval-upto $EVAL_UPTO \
    --data-root $DATA_ROOT \
    --output-root $OUTPUT_ROOT \
    --device $DEVICE \
    --num-repeat $NUM_REPEAT

# # iseg
pytest tests/perf/test_instance_segmentation.py::TestPerfInstanceSegmentation \
    --model-category all \
    --data-group $DATA_GROUP \
    --eval-upto $EVAL_UPTO \
    --data-root $DATA_ROOT \
    --output-root $OUTPUT_ROOT \
    --device $DEVICE \
    --num-repeat $NUM_REPEAT

# # sseg
pytest tests/perf/test_semantic_segmentation.py::TestPerfSemanticSegmentation \
    --model-category all \
    --data-group $DATA_GROUP \
    --eval-upto $EVAL_UPTO \
    --data-root $DATA_ROOT \
    --output-root $OUTPUT_ROOT \
    --device $DEVICE \
    --num-repeat $NUM_REPEAT

# # ano
pytest tests/perf/test_anomaly.py::TestPerfAnomalyClassification \
    --model-category all \
    --data-group $DATA_GROUP \
    --eval-upto $EVAL_UPTO \
    --data-root $DATA_ROOT \
    --output-root $OUTPUT_ROOT \
    --device $DEVICE \
    --num-repeat $NUM_REPEAT

# keypoint
pytest tests/perf/test_keypoint_detection.py::TestPerfKeypointDetection \
    --model-category all \
    --data-group $DATA_GROUP \
    --eval-upto $EVAL_UPTO \
    --data-root $DATA_ROOT \
    --output-root $OUTPUT_ROOT \
    --device $DEVICE \
    --num-repeat $NUM_REPEAT
