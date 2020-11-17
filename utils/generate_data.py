import json
import sys
import random
import copy
import argparse
import os

def generate_perturbed_data(filepath):
    lines = open(filepath, "r").readlines()
    datas = []
    for line in lines:
        data = json.loads(line)
        datas.append(data)

    output_dirpath = os.path.dirname(filepath) + "/perturbed_data/"
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    augmented_data = []
    for perturbation in ["no_context", "no_question", "no_option", "perturbed_ic"]:
        odatas = copy.deepcopy(datas)
        for item_json in odatas:
            question_text = item_json["question"]["stem"]
            last_question_text = "<s>"
            original_question_text = "<s>"

            already_added = False

            if perturbation == "no_question":
                question_text = "<s>"
                item_json["question"]["stem"] = question_text
            else:
                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    choice_label = choice_item["label"]
                    choice_text = choice_item["text"]
                    choice_context = choice_item.get("para")

                    original_question_text = question_text
                    original_choice_context = choice_context
                    original_choice_text = choice_text
                    random_float = random.random()

                    if perturbation == "perturbed_ic":
                        if item_json.get('answerKey') != choice_label and not already_added:
                            choice_text = question_text
                            choice_context = ""
                            for i in range(10):
                                choice_context += question_text + " "
                            already_added = True
                    elif perturbation == "no_context":
                        choice_context = "<s>"
                    elif perturbation == "no_option":
                        choice_text = "<s>"
                    
                    item_json["question"]["choices"][choice_id]["text"] = choice_text
                    item_json["question"]["choices"][choice_id]["para"] = choice_context

        ofile = open(output_dirpath + perturbation + ".jsonl", "w")
        for data in odatas:
            ofile.write(json.dumps(data) + "\n")

def generate_augmented_data(filepath, num_epochs):
    lines = open(filepath, "r").readlines()
    datas = []
    for line in lines:
        data = json.loads(line)
        datas.append(data)

    output_dirpath = os.path.dirname(filepath) + "/augmented_data/"
    if not os.path.exists(output_dirpath):
        os.makedirs(output_dirpath)

    # augmented_data = []

    # odatas = copy.deepcopy(datas)
    # for item_json in odatas:
    #     item_json["epoch_num"] = -1
    # augmented_data += odatas
    # print(len(augmented_data))
    ofile = open(output_dirpath + "train.jsonl", "w")
    for data in datas:
        ofile.write(json.dumps(data) + "\n")

    for epoch in range(num_epochs):
        odatas = copy.deepcopy(datas)
        
        last_question_text = "<s>"
        original_question_text = "<s>"
        
        for item_json in odatas:
            question_text = item_json["question"]["stem"]

            already_added = False
            incorrect_context = "<s>"
            incorrect_text = "<s>"
            last_context = "<s>"
            last_text = "<s>"
            for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                choice_label = choice_item["label"]
                choice_context = choice_item.get("para")
                choice_text = choice_item["text"]
                if item_json.get('answerKey') != choice_label:
                    last_context = choice_context
                    last_text = choice_text
                    if incorrect_context == "<s>":
                        incorrect_context = choice_context
                        incorrect_text = choice_text
            
            correct_option_perturbed = False

            for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                choice_label = choice_item["label"]
                choice_text = choice_item["text"]
                choice_context = choice_item.get("para")

                original_question_text = question_text
                original_choice_context = choice_context
                original_choice_text = choice_text
                random_float = random.random()
                
                # Modify correct example
                if random_float <= 0.2 and item_json.get('answerKey') == choice_label:
                    random_augmentation_float = random.random()
                    correct_option_perturbed = True
                    if random_augmentation_float <= 0.33:
                        if random.random() <= 0.5:
                            choice_context = incorrect_context
                        else:
                            choice_context = "<s>"
                    elif random_augmentation_float <= 0.66:
                        if random.random() <= 0.5:
                            choice_text = incorrect_text
                        else:
                            choice_text = "<s>"
                    else:
                        # NOTE: Modifies question for all options
                        if random.random() <= 0.5:
                            question_text = last_question_text
                        else:
                            question_text = "<s>"
                elif random_float <= 0.26 and item_json.get('answerKey') != choice_label:
                    random_augmentation_float = random.random()
                    if random_augmentation_float <= 0.5:
                        if random.random() <= 0.5:
                            choice_context = last_context
                        else:
                            choice_context = "<s>"
                    else:
                        if random.random() <= 0.5:
                            choice_text = last_text
                        else:
                            choice_text = "<s>"
                
                if item_json.get('answerKey') != choice_label:
                    last_text = original_choice_text
                    last_context = original_choice_context

                item_json["question"]["choices"][choice_id]["text"] = choice_text
                item_json["question"]["choices"][choice_id]["para"] = choice_context
                
            # Modify last question to use as irrelevant question
            if random.random() < 0.25:
                last_question_text = original_question_text

            item_json["question"]["stem"] = question_text
            item_json["correct_option_perturbed"] = str(correct_option_perturbed)
            item_json["epoch_num"] = epoch

        # augmented_data += odatas
        # print(len(odatas))

        ofile = open(output_dirpath + "train_epoch" + str(epoch) + ".jsonl", "w")
        for data in odatas:
            ofile.write(json.dumps(data) + "\n")


# MAIN
random.seed(13370)

parser = argparse.ArgumentParser(description='Generate augmented and/or perturbed data files. Output files will be generated \
                                 in the respective augmented_data/perturbed_data directories inside the lowest directory in \
                                 the respective paths.')
parser.add_argument('--path', type=str, default='./',
                    help='path to the directory containing train.jsonl and dev.jsonl which will be used to create the \
                    augmented and perturbed data files')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to generate augmented data for')
parser.add_argument('--augmented_data', action='store_true')
parser.add_argument('--perturbed_data', action='store_true')

args = parser.parse_args()

if args.augmented_data:
    generate_augmented_data(args.path + "/train.jsonl", args.epochs)

if args.perturbed_data:
    generate_perturbed_data(args.path + "/dev.jsonl")
