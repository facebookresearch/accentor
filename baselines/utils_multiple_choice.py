# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function


import logging
import os
import sys
from io import open
import json
import csv
import glob
import tqdm
from typing import List
from transformers import PreTrainedTokenizer

import random


logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, question, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.question = question
        self.contexts = contexts
        self.endings = endings
        self.label = label


class InputFeatures(object):
    def __init__(self,
                 example_id,
                 choices_features,
                 label

    ):
        self.example_id = example_id
        self.choices_features = [
            {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids
            }
            for input_ids, input_mask, segment_ids in choices_features
        ]
        self.label = label


class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()


class ACCProcessor(DataProcessor):
    def __init__(self):
        self.D = [[], [], []]

        datasetfile = "arranger_input.json"

        with open(datasetfile, "r") as f:
            data = json.load(f)
            for sid in range(2):
                dt = ["train", "dev"][sid]
                for i in range(len(data[dt])):
                    d = [data[dt][i][0].lower(), data[dt][i][1].lower(), data[dt][i][2].lower(), data[dt][i][3]]
                    self.D[sid] += [d]

        sid = 2
        for fns in [["lm.input.dev.cc.txt", "lm.output.dev.cc.txt", "dev.inference.gpt2_10epoch_1e-3_fp16.json"],
                    ["lm.input.test.cc.txt", "lm.output.test.cc.txt", "test.inference.gpt2_10epoch_1e-3_fp16.json"]]:
            with open(fns[0], "r") as f:
                data = f.read().split("\n")[0:-1:2]
            data_d = data

            with open(fns[1], "r") as f:
                data = f.read()
            data = data.split("[TransformerGenerator]:")[1:]
            for i in range(len(data)):
                data[i] = data[i].split("\n")[0].strip()
            data_cc = data

            with open(fns[2], "r") as f:
                data = json.load(f)

            for i in range(len(data)):
                data[i] = data[i].split("<|response|>")
                if len(data[i]) == 1:
                    data[i] += ['']
                elif len(data[i]) > 2:
                    data[i] = ["<|response|>".join(data[i][:-2]), data[i][-1]]
                self.D[2] += [[data_d[i].strip(), data[i][1], data_cc[i].strip(), 0]]
                    

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[0], "train")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[2], "test") 

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self.D[1], "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]


    def _create_examples(self, data, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, d) in enumerate(data):
                    
            acc_id = "%s-%d" % (set_type, i)

            examples.append(
                InputExample(
                    example_id=acc_id,
                    question="",
                    contexts=[data[i][0], data[i][0], data[i][0]],
                    endings=[data[i][1], data[i][2] + " " + data[i][1], data[i][1] + " " + data[i][2]],
                    label=str(data[i][3])))
        return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_length: int,
    tokenizer: PreTrainedTokenizer,
    pad_token_segment_id=0,
    pad_on_left=False,
    pad_token=0,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """
    Loads a data file into a list of `InputFeatures`
    """

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in tqdm.tqdm(enumerate(examples), desc="convert examples to features"):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        choices_features = []
        for ending_idx, (context, ending) in enumerate(zip(example.contexts, example.endings)):
            text_a = context
            if example.question.find("_") != -1:
                text_b = example.question.replace("_", ending)
            else:
                text_b = example.question + " " + ending

            inputs = tokenizer.encode_plus(
                text_a,
                text_b,
                add_special_tokens=True,
                max_length=max_length,
            )

            input_ids, token_type_ids = inputs["input_ids"], inputs["token_type_ids"]

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                attention_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + attention_mask
                token_type_ids = ([pad_token_segment_id] * padding_length) + token_type_ids
            else:
                input_ids = input_ids + ([pad_token] * padding_length)
                attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
                token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)

            assert len(input_ids) == max_length
            assert len(attention_mask) == max_length
            assert len(token_type_ids) == max_length
            choices_features.append((input_ids, attention_mask, token_type_ids))


        label = label_map[example.label]

        if ex_index < 2:
            logger.info("*** Example ***")
            logger.info("race_id: {}".format(example.example_id))
            for choice_idx, (input_ids, attention_mask, token_type_ids) in enumerate(choices_features):
                logger.info("choice: {}".format(choice_idx))
                logger.info("input_ids: {}".format(' '.join(map(str, input_ids))))
                logger.info("attention_mask: {}".format(' '.join(map(str, attention_mask))))
                logger.info("token_type_ids: {}".format(' '.join(map(str, token_type_ids))))
                logger.info("label: {}".format(label))

        features.append(
            InputFeatures(
                example_id=example.example_id,
                choices_features=choices_features,
                label=label,
            )
        )

    return features




processors = {
    "acc": ACCProcessor,
}


MULTIPLE_CHOICE_TASKS_NUM_LABELS = {
    "acc", 3
}
