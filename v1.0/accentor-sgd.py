# Copyright (c) Facebook, Inc. and its affiliates.

import json
import argparse
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="./dstc8-schema-guided-dialogue", type=str, required=False, help="Path to the SGD dataset.")
    parser.add_argument("--target", default="./accentor-sgd", type=str, required=False, help="The target directory to store ACCENTOR-SGD.")
    args = parser.parse_args()

    with open("candidates-sgd.json", "r", encoding='utf8') as f:
        augmentation = json.load(f)

    for subdir in ["train", "dev", "test"]:
        targetdir = os.path.join(args.target, subdir)
        sourcedir = os.path.join(args.source, subdir)
        os.makedirs(targetdir, exist_ok=True)
        fns = os.listdir(sourcedir)
        for fn in fns:
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(sourcedir, fn), "r", encoding='utf8') as f:
                data = json.load(f)
            if fn.startswith("dialogue"):
                for i in range(len(data)):
                    for j in range(1, len(data[i]["turns"]), 2):
                        data[i]["turns"][j]["beginning"] = []
                        data[i]["turns"][j]["end"] = []
                    for cc in augmentation[subdir + data[i]["dialogue_id"]]:
                        data[i]["turns"][cc[0]][cc[1]] += [{"candidate": cc[2], "label": cc[3], "justification": cc[4]}]
            with open(os.path.join(targetdir, fn), "w", encoding='utf8') as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
            
            
