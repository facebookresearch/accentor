# Copyright (c) Facebook, Inc. and its affiliates.

import json

with open("./acc_arranger_roberta_base_3epoch/is_test_true_eval_logits.txt", "r") as f:
    model_outputs = f.read().strip().split("\n")
    for i in range(len(model_outputs)):
        model_outputs[i] = model_outputs[i].split()
        for j in range(len(model_outputs[i])):
            model_outputs[i][j] = float(model_outputs[i][j])
        assert(len(model_outputs[i]) == 3)
    print(len(model_outputs))

for fns in [["./lm.input.dev.cc.txt", "./lm.output.dev.cc.txt", "./dev.inference.gpt2_10epoch_1e-3_fp16.json", "./dev.inference.arranger_3epoch.json"],
            ["./lm.input.test.cc.txt", "./lm.output.test.cc.txt", "./test.inference.gpt2_10epoch_1e-3_fp16.json", "./test.inference.arranger_3epoch.json"]]:
    with open(fns[0], "r") as f:
        data = f.read().split("\n")[0:-1:2]
    print(len(data))
    data_d = data

    with open(fns[1], "r") as f:
        data = f.read()
    data = data.split("[TransformerGenerator]:")[1:]
    for i in range(len(data)):
        data[i] = data[i].split("\n")[0].strip()
    print(len(data))
    data_cc = data

    with open(fns[2], "r") as f:
        data = json.load(f)
    print(len(data))

    eval_data = []
    for i in range(len(data)):
        data[i] = data[i].split("<|response|>")
        if len(data[i]) == 1:
            data[i] += ['']
        elif len(data[i]) > 2:
            data[i] = ["<|response|>".join(data[i][:-2]), data[i][-1]]
        eval_data += [[data_d[i].strip(), data[i][1], data_cc[i].strip(), 0]]

    print(len(eval_data))

    stats = {0:0, 1:0, 2:0}
    for i in range(len(data)):
        assert(len(model_outputs[i]) == 3)
        o = 0
        for j in range(1, 3):
            if model_outputs[i][j] > model_outputs[i][o]:
                o = j
        stats[o] += 1
        if o == 0:
            data[i] = "<|response|>".join(data[i])
        elif o == 1:
            data[i] = data[i][0] + "<|response|> " + data_cc[i].strip() + " " + data[i][1].strip()
        else:
            data[i] = data[i][0] + "<|response|> " + data[i][1].strip() + " " + data_cc[i].strip()

    print(len(data), len(model_outputs))
    print(stats)
    model_outputs = model_outputs[len(data):]

    with open(fns[3], "w", encoding='utf8') as f:
        json.dump(data, f, indent=1)
