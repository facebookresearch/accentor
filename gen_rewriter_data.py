# Copyright (c) Facebook, Inc. and its affiliates.

import json

for fns in [["./lm.input.dev.eval.txt", "./lm.output.dev.cc.txt", "./dev.inference.gpt2_10epoch_1e-3_fp16.json", "lm.input.dev.eval.ff.txt"],
            ["./lm.input.test.eval.txt", "./lm.output.test.cc.txt", "./test.inference.gpt2_10epoch_1e-3_fp16.json", "lm.input.test.eval.ff.txt"]]:
    with open(fns[0], "r", encoding='utf8') as f:
        context = f.read().strip().split("\n")
    with open(fns[1], "r", encoding='utf8') as f:
        cc = f.read().strip()
        cc = cc.split("[TransformerGenerator]:")[1:]
        for i in range(len(cc)):
            cc[i] = cc[i].split("\n")[0].strip()
    with open(fns[2], "r", encoding='utf8') as f:
        task = json.load(f)
    print(len(context), len(cc), len(task))
    assert(len(context) == len(cc))
    assert(len(cc) == len(task))
    with open(fns[3], "w", encoding='utf8') as f:
        for i in range(len(cc)):
            t = task[i].split("<|response|>")
            if len(t) >= 2:
                t = t[-1].strip()
            else:
                t = ""
            b = task[i].split("<|belief|>")
            if len(b) >= 2:
                b = b[1].split("<|endofbelief|>")
                if len(b) == 2:
                    b = b[0]
                else:
                    b = ""
            else:
                b = ""
            f.write(context[i] + " <|task|> " + t + " <|endoftask|> <|chitchat|> " + cc[i] + ' <|endofchitchat|> <|belief|>' + b + "<|endofbelief|>\n")
        
        
