// Variables to be set externally
// DATASET ({race, qasc, aristo})
// TRAIN_MODE ({proposed, baseline, binary})
// RACE_BASELINE_MODEL: path to model to be starting point of training (RACE Baseline model)
// {RACE, QASC, ARISTO}_DATA_PATH: paths to the respective dataset directories,
//                                 refer to preprocessing step for further details

local utils = import 'utils.libsonnet';

local dataset = std.extVar("DATASET");
local mode = std.extVar("TRAIN_MODE");
local datadir = std.extVar(std.asciiUpper(dataset) + "_DATA_PATH");
local augdatadir = datadir + "/augmented_data";

// Tunable parameters
local epochs = if mode == "proposed" then 5 else 4;
local max_pieces = if dataset == "race" then 384 else 256;
local num_choices = if dataset == "qasc" then 8 else 4;
local optimizer_weight_decay = if dataset == "race" then 0.01 else 0.1;

// Implementation specific parameters
local lazy = if mode == "proposed" then true else false;
local shuffle_dataset = if mode == "baseline" then false else true;
local gpu = if std.extVar("GPU") == "-1" then -1 else 0;

// Training mode/dataset related parameters
local augmentation_since_epoch = if mode == "proposed" then 0 else -1;
local binary_loss = if mode == "baseline" then false else true;
local weights_model = if dataset == "race" then null else std.extVar("RACE_BASELINE_MODEL");

{
    "dataset_reader": {
        "type": "transformer_mc_qa",
        "training": true,
        "augmentation_since_epoch": augmentation_since_epoch,
        "lazy": lazy,
        "shuffle_dataset": shuffle_dataset,
        "num_choices": num_choices,
        "max_pieces": max_pieces,
        "pretrained_model": "roberta-large"
    },
    "data_loader": {
        "sampler": 
            if shuffle_dataset then null
            else {
                "type": "shuffle_sampler"
            }
            ,
        "batch_size": 1
    },
    "model": {
        "type": "roberta_mc_qa",
        "pretrained_model": "roberta-large",
        "binary_loss": binary_loss,
        "transformer_weights_model": weights_model
    },
    "train_data_path":
        if mode == "proposed" then augdatadir + "/train.jsonl"
        else datadir + "/train.jsonl"
    ,
    "validation_data_path": datadir + "/dev.jsonl",
    "test_data_path": datadir + "/test.jsonl",
    "trainer": {
        "cuda_device": gpu,
        "grad_clipping": 1,
        "num_gradient_accumulation_steps": 16,
        "learning_rate_scheduler": {
            "type": "slanted_triangular",
            "cut_frac": 0.06,
            "num_epochs": epochs,
            "num_steps_per_epoch": 
                if dataset == "aristo" then 562
                else if dataset == "race" then 5492
                else if dataset == "qasc" then 509
        },
        "num_epochs": epochs,
        "optimizer": {
            "type": "adamw",
            "betas": [
                0.9,
                0.98
            ],
            "lr": 1e-05,
            "parameter_groups": [
                [
                    [
                        "bias",
                        "LayerNorm\\.weight",
                        "layer_norm\\.weight"
                    ],
                    {
                        "weight_decay": 0
                    }
                ]
            ],
            "weight_decay": 0.1
        },
        "validation_metric": "+accuracy"
    },
    "datasets_for_vocab_creation": [],
    "evaluate_on_test": false
}
