# What do we expect from Multiple-choice QA Systems?
This is the official repository for the Findings of EMNLP 2020 paper, "[What do we expect from Multiple-choice QA Systems?](https://www.aclweb.org/anthology/2020.findings-emnlp.317/)".

## Resources
This repository uses data from the [Aristo](https://arxiv.org/abs/1909.01958), [QASC](https://arxiv.org/abs/1910.11473) (data retrieved using two-step approach) and [RACE](https://arxiv.org/abs/1704.04683) datasets. The QASC dataset can be directly downloaded from [qasc](http://data.allenai.org/downloads/qasc/qasc_dataset_2step.zip) while the RACE dataset can be obtained by filling the form at [race](https://www.cs.cmu.edu/~glai1/data/race/). Kindly contact the authors of the Aristo paper for access to their dataset.

Trained models coming soon.

## Environment
This repository is tested on Python 3.7.9. 

The following commands create a conda environment and install the requisite dependencies
```
# Create and activate conda environment
conda create --name mcqa-expectations python=3.7
conda activate mcqa-expectations

# Clone repository
git clone https://github.com/CogComp/mcqa-expectations.git
cd mcqa-expectations

# Install packages
(torch1.6.0 install command)
pip install -r requirements.txt
```
Replace the `(torch1.6.0 install command)` with the appropriate command from [torch install](https://pytorch.org/get-started/previous-versions/) according to the available version of CUDA. The results were reported using CUDA Version 10.1.168.

The following commands must to be run in the `mcqa-expectations` environment.

## Preliminaries

For efficiency purposes, kindly `export` the following variables to the respective absolute paths, i.e relative to the root directory

- `ARISTO_DATA_PATH, QASC_DATA_PATH`: directories containing the respective `{train, dev, test}.jsonl` files
- `RACE_DATA_PATH`: directory created by the commmand `tar -zxvf RACE.tar.gz`. 
- `OUPUT_MODELS_DIR`: directory where trained models are to be stored and where all previously trained models are assumed to be present for evaluation
- `OUPUT_EVAL_DIR`: directory where evaluation output jsons are to be stored 
- `GPU`: to the appropriate gpu device or -1 if none


## Preprocessing

### Dataset formatting
The code assumes the files to be in [json lines](https://jsonlines.org) format where each datapoint is of the form:
```
{
	"id": ...,
	"answerKey": ...,
	"question": {
		"stem": ...,
		"choices": [
			{
				"label": ...,
				"text": ...,
				"para": ...,
			}, ...
		]
	}
}
```

Run the command `bash preprocess.sh` to generate the data required by the training and evaluation steps.


## Training
The following command is used to train the respective models
```
bash train.sh (dataset) (train_mode)
```
where `(dataset)` is one of `{race, qasc, aristo}` and `(train_mode)` is one of `{baseline, proposed, binary}`. The output model is stored as `$(OUPUT_MODELS_DIR)/(dataset)_(train_mode)`.

Training any Aristo or QASC model assumes the presence of `$(OUPUT_MODELS_DIR)/race_baseline` model since all Aristo and QASC models use the baseline finetuning of RACE as the starting point. 

Meaning of the different taining modes:

- `baseline`: trains a simple RoBERTa model with a classification layer on the respective dataset
- `proposed`: trains the RoBERTa model on the respective dataset with our proposed modifications to the training approach. This includes changing the loss to a binary classification loss and using unsupervised data augmentation.
- `binary`: same as `baseline` except that it uses binary classification loss instead of the cross entropy loss used in the baseline training.

## Evaluation
The following command is used to evaluate models
```
bash evaluate.sh (dataset) (trained_mode) (eval_mode)
```
- `(dataset)` is one of `{race, qasc, aristo}`
- `(trained_mode)` is one of `{baseline, proposed, binary}` which is used to find the location of the trained model as `OUPUT_MODELS_DIR/(dataset)_(trained_mode)` maintained by the training command.
- `(eval_mode)` is one of `{dev, test, perturbed_ic, no_context, no_option, no_question}`. Kindly refer to the paper for further details on the evaluation settings.

## References

Please consider citing our work if you found it helpful to your research
```
@inproceedings{ShahGuRo20,
    author = {Krunal Shah and Nitish Gupta and Dan Roth},
    title = {{What Do We Expect from Multiple-Choice QA Systems?}},
    booktitle = {Findings of the Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    year = {2020},
    url = "https://cogcomp.seas.upenn.edu/papers/ShahGuRo20.pdf",
}
```

## Contributions and contact
This code was developed by [Krunal Shah](https://github.com/krunal-shah) on top of [Oyvind Tafjord's](https://github.com/OyvindTafjord) AllenNLP fork, contact [ktgshah@gmail.com](ktgshah@gmail.com) for support.

If you'd like to contribute code, feel free to open a [pull request](https://github.com/CogComp/mcqa-expectations/pulls). If you find an issue with the code or require additional support, please open an [issue](https://github.com/CogComp/mcqa-expectations/issues).
