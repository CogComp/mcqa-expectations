# What do we expect from Multiple-choice QA Systems?
Official PyTorch repository for the paper "What do we expect from Multiple-choice QA Systems?"

## Resources
This repository uses data from the [Aristo](https://arxiv.org/abs/1909.01958), [QASC](https://arxiv.org/abs/1910.11473) (data retrieved using two-step approach) and [RACE](https://arxiv.org/abs/1704.04683) datasets. The QASC dataset can be directly downloaded from [qasc](http://data.allenai.org/downloads/qasc/qasc_dataset_2step.zip) while the RACE dataset can be obtained by filling the form at [race](https://www.cs.cmu.edu/~glai1/data/race/). Kindly contact the authors for access to the Aristo dataset.

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
Replace the `(torch1.6.0 install command)` with the appropriate command from [torch install](https://pytorch.org/get-started/locally/) according to the available version of CUDA. The results were reported using CUDA Version 10.1.168.

The following commands must to be run in the `mcqa-expectations` environment.

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

For efficiency purposes, kindly set the following variables as
```
export RACE_DATA_PATH="(race-data-path)"
export QASC_DATA_PATH="(qasc-data-path)"
export ARISTO_DATA_PATH="(aristo-data-path)"
```
where the directories pointed by `(qasc-data-path)` and `(aristo-data-path)` contain the respective `{train, dev, test}.jsonl` files. The path `(race-data-path)` points to the directory created by the commmand `tar -zxvf RACE.tar.gz`.

Run the command `bash preprocess.sh` to generate the data required by the training and evaluation steps.

## Training
The `configs/` directory contains the following configuration files for training the respective models

- `{race, qasc, aristo}_baseline.json`: Train a simple RoBERTa model with a classification layer on the respective dataset
- `{race, qasc, aristo}_proposed.json`: Train the RoBERTa model on the respective dataset with our proposed modifications to the training approach. This includes chaing the loss to a binary classification loss and using unsupervised data augmentation.
- `{race, qasc, aristo}_binary.json`: 


## Evaluation
`allennlp `