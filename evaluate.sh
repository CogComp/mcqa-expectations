export DATASET=$1
export TRAIN_MODE=$2
export EVAL_MODE=$3
export GPU="${GPU:-0}"
export DATA_FILE_PATH="${DATASET^^}"_DATA_PATH

if [[ $3 == "dev" || $3 == "test" ]]
then
  export FILENAME=$3.jsonl
else
  export FILENAME="perturbed_data/$3.jsonl"
fi

# allennlp train -s ${OUPUT_MODELS_DIR}/$1_$2 train_config.jsonnet
allennlp evaluate --cuda-device ${GPU} --output-file ${OUPUT_EVAL_DIR}/$1_$2_$3.json ${OUPUT_MODELS_DIR}/$1_$2 ${!DATA_FILE_PATH}/$FILENAME

# datadir + "/dev.jsonl",

# local datadir = std.extVar(std.asciiUpper(dataset) + "_DATA_PATH");