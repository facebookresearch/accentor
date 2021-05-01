# Copyright (c) Facebook, Inc. and its affiliates.

import json
import random
import argparse
import os

def clean(x):
    return x.replace("\n", "").replace("\r", "").replace("\t", " ").strip()

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./accentor-sgd/", type=str, required=False, help="path to SGD")
args = parser.parse_args()

random.seed(42)

pairs = {}
for s in ["train", "dev", "test"]:
    pairs[s] = []
    fns = os.listdir(args.data + s)
    fns.sort()
    for fn in fns:
        if not fn.startswith("dialogue") or not fn.endswith(".json"):
            continue
        with open(args.data + s + "/" + fn, "r", encoding='utf8') as f:
            data = json.load(f)
        for i in range(len(data)):
            t = ''
            for j in range(len(data[i]["turns"])):
                for ps in ["beginning", "end"]:
                    if ps in data[i]["turns"][j]:
                        for k in range(len(data[i]["turns"][j][ps])):
                            if data[i]["turns"][j][ps][k]["label"] == "good":
                                pair = [t, clean(data[i]["turns"][j][ps][k]["candidate"])]
                                pairs[s] += [pair]
                if t != '':
                    t += ' '
                if j % 2 == 0:
                    t += 'user: '
                else:
                    t += 'system: '
                t += clean(data[i]["turns"][j]["utterance"])

for s in pairs:
    print(s, len(pairs[s]))

random.shuffle(pairs["train"])

for s in ["train", "dev", "test"]:
    with open("parlai_"+(s if s != "dev" else "valid")+".txt", "w", encoding='utf8') as f:
        for i in range(len(pairs[s])):
            f.write("text:" + pairs[s][i][0] + "\t" + "labels:" + pairs[s][i][1] + "\tepisode_done:True\n")



