export DATASET=$1
export TRAIN_MODE=$2
export RACE_BASELINE_MODEL=${OUTPUT_MODELS_DIR}/race_baseline/
allennlp train -s ${OUPUT_MODELS_DIR}/$1_$2 train_config.jsonnet
# python generate_config.py train_config.jsonnet ${OUTPUT_MODELS_DIR}/$1_$2.json