import argparse
import json
import sys
import os
import pdb

def generate_files(path):
    datas = []
    path = path.strip()
    path = path[:-1] if path[-1] == '/' else path 
    output = open(path + ".jsonl", "w")
    for prefix in ["high", "middle"]:
        dirpath = path + "/" + prefix + "/"
        for filename in os.listdir(dirpath):
            if not filename.endswith(".txt"):
                continue
            lines = open(dirpath + filename, "r").readlines()
            data = json.loads(lines[0])
            for i in range(len(data["questions"])):
                ndata = {}
                assert len(data["options"][i]) == 4
                ndata["id"] = data["id"].replace(".", "_" + str(i) + "_.")
                ndata["question"] = {}
                ndata["answerKey"] = data["answers"][i]
                ndata["question"]["stem"] = data["questions"][i]
                ndata["question"]["choices"] = []
                article = data["article"]
                for j in range(4):
                    ndata["question"]["choices"].append({"label":chr(ord('A') + j) , "para": article, "text": data["options"][i][j]})
                datas.append(ndata)

    for data in datas:
        output.write(json.dumps(data)+ "\n")

    return path + ".jsonl"

def process(fpath, alignment_path):
    lines = open(fpath, "r").readlines()
    datas = []
    for line in lines:
        datas.append(json.loads(line))
    alignments = open(alignment_path, "r").readlines()
    alignment = {}
    for i, idx in enumerate(alignments):
        alignment[idx.strip()] = i
    datas = sorted(datas, key = lambda data: alignment[data["id"]])
    ofile = open(fpath, "w")
    for data in datas:
        ofile.write(json.dumps(data) + "\n")

# MAIN
parser = argparse.ArgumentParser(description='Used for generating (aligned) single train/test/dev files for RACE or for aligning \
                                              the files for QASC and Aristo to the files which used for reported results.')
parser.add_argument('--path', type=str, default="./data/",
                    help='path to the aristo or qasc directory with {train, test, dev}.jsonl files withing, or path to the highest \
                         level directory for RACE dataset')
parser.add_argument('--dataset', type=str, default="RACE",
                    help='which dataset is being processed {RACE, QASC_2Step, Aristo}')
parser.add_argument('--alignment_path', type=str, default="./alignment_ids/",
                    help='path to the alignment files directory')

args = parser.parse_args()

splits = ["train", "dev", "test"]
for split in splits:
    if args.dataset == 'race':
        ofilename = generate_files(args.path + "/" + split)
    else:
        ofilename = args.path + "/" + split + ".jsonl"
    process(ofilename, args.alignment_path + "/" + args.dataset + "_" + split + ".txt")
