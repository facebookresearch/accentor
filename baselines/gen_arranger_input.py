# Copyright (c) Facebook, Inc. and its affiliates.

import json
import random
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--data", default="./simpletod/", type=str, required=False, help="path to delexed & augmented SGD")
args = parser.parse_args()

def clean(x):
    return x.replace("\n", "").replace("\r", "").replace("\t", " ").strip()

random.seed(42)

pairs = {}
pos = {}
tot = {}

for s in ["train", "dev", "test"]:
    pairs[s] = []
    pos[s] = 0
    tot[s] = 0
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
                                tot[s] += 1
                                if data[i]["turns"][j][ps][k]["label"] == "good":
                                    pair = [t, data[i]["turns"][j]["delex"], clean(data[i]["turns"][j][ps][k]["candidate"]), 1 if ps == "beginning" else 2]
                                    pairs[s] += [pair]
                                    pos[s] += 1
                                else:
                                    pair = [t, data[i]["turns"][j]["delex"], clean(data[i]["turns"][j][ps][k]["candidate"]), 0]
                                    pairs[s] += [pair]
                    if t != '':
                        t += ' '
                    if j % 2 == 0:
                        t += 'user: '
                    else:
                        t += 'system: '
                    t += clean(data[i]["turns"][j]["utterance"])

for s in pos:
    print(s, pos[s], tot[s], pos[s]/tot[s])

for s in pairs:
    print(s, len(pairs[s]))

random.shuffle(pairs["train"])

with open("arranger_input.json", "w", encoding='utf8') as f:
    json.dump(pairs, f, indent=1)



