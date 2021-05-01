# Copyright (c) Facebook, Inc. and its affiliates.

import json
import os
from utils import bleuscorer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference", default="dev.inference.gpt2_10epoch_1e-3_fp16.json", type=str, required=False, help='inference file')
    parser.add_argument("--datafolder", default="./simpletod/", type=str, required=False, help='data folder')
    parser.add_argument("--predictionfolder", default="./prediction/", type=str, required=False, help='prediction folder')
    parser.add_argument("--split", default="dev", type=str, required=False, help="[dev,test]")
    args = parser.parse_args()
    inference = args.inference    
    datafolder = args.datafolder
    predictionfolder = args.predictionfolder
    folder = args.split + "/"

    if inference.endswith(".txt"):
        with open(inference, "r") as f:
            predict = f.read().strip().split("\n")
            predict = [a.strip() for a in predict]
    else:
        with open(inference, "r") as f:
            predict = json.load(f)

    idx = 0
    cnt = 0

    seen_services = set()
    with open(datafolder + "train/" + "schema.json", "r") as f:
        schema = json.load(f)
        for i in range(len(schema)):
            seen_services.add(schema[i]["service_name"])

    domain_slots = set()
    with open(datafolder + folder + "schema.json", "r") as f:
        schema = json.load(f)
        for i in range(len(schema)):
            for j in range(len(schema[i]["slots"])):
                assert(" " not in schema[i]["slots"][j])
                domain_slots.add(schema[i]["service_name"].split("_")[0].lower() + " " + schema[i]["slots"][j]["name"].lower())


    fns = os.listdir(datafolder + folder)
    fns.sort()

    act_precision = []
    act_recall = []
    seen_act_precision = []
    seen_act_recall = []
    unseen_act_precision = []
    unseen_act_recall = []
    bleu = []
    bleua = []
    bleub = []
    seenbleu = []
    seenbleua = []
    seenbleub = []
    unseenbleu = []
    unseenbleua = []
    unseenbleub = []

    for fn in fns:
        if not fn.startswith("dialogue"):
            continue
        if fn.startswith("dialogues_and_metrics.json"):
            continue
        with open(datafolder + folder + fn, "r") as f:
            data = json.load(f)

        for i in range(len(data)):
            for j in range(1, len(data[i]["turns"]), 2):
                cnt += 1
                if idx >= len(predict):
                    continue
                belief = predict[idx].split("<|belief|>")
                if len(belief) >= 2 and "<|endofbelief|>" in belief[1]:
                    belief = belief[1].split("<|endofbelief|>")[0].strip()
                else:
                    belief = ""
                action = predict[idx].split("<|action|>")
                if len(action) >= 2 and "<|endofaction|>" in action[1]:
                    action = action[1].split("<|endofaction|>")[0].strip()
                else:
                    action = ""
                response = predict[idx].split("<|response|>")
                if len(response) >= 2:
                    response = response[1].split("<|")[0].strip()
                else:
                    response = ""
                data[i]["turns"][j]["response"] = response

                seen = True
                for k in range(len(data[i]["turns"][j-1]["frames"])):
                    if data[i]["turns"][j-1]["frames"][k]["service"] not in seen_services:
                        seen = False

                parsedbelief = belief.split(", ")
                for k in range(len(parsedbelief)):
                    parsed = False
                    for ds in domain_slots:
                        if parsedbelief[k].startswith(ds):
                            parsedbelief[k] = [ds, parsedbelief[k][len(ds):].strip()]
                            parsed = True
                            break
                    if not parsed:
                        parsedbelief[k] = [parsedbelief[k]]
                k = 1
                while k < len(parsedbelief):
                    if len(parsedbelief[k]) == 1:
                        parsedbelief[k-1] += parsedbelief[k]
                        del parsedbelief[k]
                    else:
                        k += 1
                if len(parsedbelief) >= 1:
                    if parsedbelief[0][0] not in domain_slots:
                        del parsedbelief[0]
                parsedbelief = {x[0]:x[1:] for x in parsedbelief}

                parsedaction = action.split(", ")
                for k in range(len(parsedaction)):
                    parsedaction[k] = parsedaction[k].strip().split()
                k = 0
                while k < len(parsedaction):
                    if len(parsedaction[k]) <= 1 or len(parsedaction[k]) > 3:
                        del parsedaction[k]
                    else:
                        k += 1
                act_gt = set()
                for k in range(len(data[i]["turns"][j]["frames"][0]["actions"])):
                    act_gt.add((data[i]["turns"][j]["frames"][0]["actions"][k]["act"].lower() + " " + data[i]["turns"][j]["frames"][0]["actions"][k]["slot"]).strip())
                act_p = set()
                for k in range(len(parsedaction)):
                    act_p.add(' '.join(parsedaction[k][1:]))

                act_precision += [len(act_p & act_gt) / len(act_p) if len(act_p) != 0 else 1]
                act_recall += [len(act_p & act_gt) / len(act_gt) if len(act_gt) != 0 else 0]
                if seen:
                    seen_act_precision += [len(act_p & act_gt) / len(act_p) if len(act_p) != 0 else 1]
                    seen_act_recall += [len(act_p & act_gt) / len(act_gt) if len(act_gt) != 0 else 0]
                else:
                    unseen_act_precision += [len(act_p & act_gt) / len(act_p) if len(act_p) != 0 else 1]
                    unseen_act_recall += [len(act_p & act_gt) / len(act_gt) if len(act_gt) != 0 else 0]

                bleu += [bleuscorer([response.lower()], [[data[i]["turns"][j]["delex"].lower()]])]

                if len(data[i]["turns"][j]["delexaug"]) > 0:
                    bleua += [bleuscorer([response.lower()], [[a.lower() for a in data[i]["turns"][j]["delexaug"]]])]
                bleub += [bleuscorer([response.lower()], [[a.lower() for a in data[i]["turns"][j]["delexaug"] + [data[i]["turns"][j]["delex"].lower()]]])]

                if seen:
                    seenbleu += [bleuscorer([response.lower()], [[data[i]["turns"][j]["delex"].lower()]])]
                    if len(data[i]["turns"][j]["delexaug"]) > 0:
                        seenbleua += [bleuscorer([response.lower()], [[a.lower() for a in data[i]["turns"][j]["delexaug"]]])]
                    seenbleub += [bleuscorer([response.lower()], [[a.lower() for a in data[i]["turns"][j]["delexaug"] + [data[i]["turns"][j]["delex"].lower()]]])]
                else:
                    unseenbleu += [bleuscorer([response.lower()], [[data[i]["turns"][j]["delex"].lower()]])]
                    if len(data[i]["turns"][j]["delexaug"]) > 0:
                        unseenbleua += [bleuscorer([response.lower()], [[a.lower() for a in data[i]["turns"][j]["delexaug"]]])]
                    unseenbleub += [bleuscorer([response.lower()], [[a.lower() for a in data[i]["turns"][j]["delexaug"] + [data[i]["turns"][j]["delex"].lower()]]])]

                for k in range(len(data[i]["turns"][j-1]["frames"])):
                    data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"] = {}
                    for ds in parsedbelief:
                        if ds.split()[0].lower() == data[i]["turns"][j-1]["frames"][k]["service"].split("_")[0].lower():
                            data[i]["turns"][j-1]["frames"][k]["state"]["slot_values"][ds.split()[1]] = parsedbelief[ds]
                idx += 1

        if not os.path.exists(predictionfolder + folder):
            os.makedirs(predictionfolder + folder)
        with open(predictionfolder + folder + fn, "w") as f:
            json.dump(data, f, indent=1)

    act_precision = sum(act_precision) / len(act_precision)
    act_recall = sum(act_recall) / len(act_recall)
    print("act", act_precision, act_recall, 2*act_precision*act_recall/(act_precision+act_recall))
    print('bleu:', sum(bleu)/len(bleu)) #BLEU-4_{orig}
    print('bleua:', sum(bleua)/len(bleua)) #BLEU-4_{aug}
    #print('bleub:', sum(bleub)/len(bleub))
    seen_act_precision = sum(seen_act_precision) / len(seen_act_precision)
    seen_act_recall = sum(seen_act_recall) / len(seen_act_recall)
    print("act (seen):", seen_act_precision, seen_act_recall, 2*seen_act_precision*seen_act_recall/(seen_act_precision+seen_act_recall))
    unseen_act_precision = sum(unseen_act_precision) / len(unseen_act_precision)
    unseen_act_recall = sum(unseen_act_recall) / len(unseen_act_recall)
    print("act (unseen):", unseen_act_precision, unseen_act_recall, 2*unseen_act_precision*unseen_act_recall/(unseen_act_precision+unseen_act_recall))
    print('bleu (seen):', sum(seenbleu)/len(seenbleu))
    print('bleua (seen):', sum(seenbleua)/len(seenbleua))
    #print('bleub (seen):', sum(seenbleub)/len(seenbleub))
    print('bleu (unseen):', sum(unseenbleu)/len(unseenbleu))
    print('bleua (unseen):', sum(unseenbleua)/len(unseenbleua))
    #print('bleub (unseen):', sum(unseenbleub)/len(unseenbleub))
    
if __name__ == '__main__':
    main()
