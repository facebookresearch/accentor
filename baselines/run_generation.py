# Copyright (c) Facebook, Inc. and its affiliates.

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import argparse
import numpy as np
import json
from tqdm import tqdm

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

parser = argparse.ArgumentParser()
parser.add_argument("--no_cuda", action="store_true", help="avoid using CUDA when available")
parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
parser.add_argument("--model_name_or_path", type=str, default="output", help="path to pre-trained model or shortcut name")
parser.add_argument("--input", type=str, help="input text file, each line corresponding to one instance")
parser.add_argument("--output", type=str, help="output file")
parser.add_argument("--eos_token_id", type=int, default=None, help="eos token id")
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument("--jobid", type=int, default=0, help="jobid")
parser.add_argument("--jobnum", type=int, default=1, help="jobnum")

args = parser.parse_args()
args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

set_seed(args)

model = GPT2LMHeadModel.from_pretrained(args.model_name_or_path)
tokenizer = GPT2Tokenizer.from_pretrained(args.model_name_or_path, pad_token='<PAD>')

model.to(args.device)

with open(args.input, "r") as f:
    prompts = f.read().strip().split("\n")

batch_size = args.batch_size

ret = []

for batch in tqdm(range(args.jobid, len(prompts), batch_size * args.jobnum)):

    prompt_text = prompts[batch: batch+batch_size]

    encodings_dict = tokenizer.batch_encode_plus(prompt_text, max_length=None, pad_to_max_length=True)

    input_ids = torch.tensor(encodings_dict['input_ids'])
    attn_mask = torch.tensor(encodings_dict['attention_mask'])

    seq_len = len(input_ids[0])

    num_tokens_to_produce = 1024 - seq_len
    pad_token_id = tokenizer.pad_token_id
    eos_token_id = args.eos_token_id
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    eos_not_in_sents = torch.ones(input_ids.shape[0]).long()

    last_non_masked_idx = torch.sum(attn_mask, dim=1) - 1
    start_idx = inp_idx = (last_non_masked_idx).view(-1, 1).repeat(1, tokenizer.vocab_size + len(tokenizer.additional_special_tokens)).unsqueeze(1)
    past = None

    position_ids = torch.tensor([list(range(seq_len)) for i in range(input_ids.shape[0])])
    for i, position_ids_slice in enumerate(position_ids):
        position_ids_slice[last_non_masked_idx[i]:] = position_ids_slice[last_non_masked_idx[i]]

    input_ids = input_ids.to(args.device)
    attn_mask = attn_mask.to(args.device)
    eos_not_in_sents = eos_not_in_sents.to(args.device)
    start_idx = start_idx.to(args.device)
    position_ids = position_ids.to(args.device)


    for step in range(num_tokens_to_produce):
        outputs = model(input_ids, attention_mask=attn_mask, position_ids=position_ids)

        if step == 0:
            next_token_logits = outputs[0].gather(1, start_idx).squeeze(1)
        else:
            next_token_logits = outputs[0][:, -1, :]

        next_tokens = torch.argmax(next_token_logits, dim=-1)

        eos_not_in_sents.mul_(next_tokens.ne(eos_token_id).long())

        tokens_to_add = next_tokens * (eos_not_in_sents) + pad_token_id * (1 - eos_not_in_sents)

        input_ids = torch.cat([input_ids, tokens_to_add.unsqueeze(-1)], dim=-1)
        attn_mask = torch.cat([attn_mask, torch.ones((attn_mask.shape[0], 1)).long().to(args.device)], dim=1)
        position_ids = torch.cat([position_ids, (position_ids[:, -1] + 1).unsqueeze(-1)], dim=1)

        if torch.max(eos_not_in_sents) == 0:
            break

    ret += [tokenizer.decode(output, skip_special_tokens=False, clean_up_tokenization_spaces=True).replace("<|endoftext|>", "") for output in input_ids]

with open(args.output, "w") as f:
    json.dump(ret, f, indent=1)
