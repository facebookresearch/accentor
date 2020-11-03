# ACCENTOR: Adding Chit-Chats to Enhance Task-Oriented Dialogues

## Overview

ACCENTOR consists of the human-annotated chit-chat additions to the 23.8K dialogues from Schema Guided Dialogue (SGD) and MultiWOZ 2.1, allowing researchers to study contexutal addition of chit-chat utterances for virtual assistants, to make task-oriented dialogues more engaging and social. 

We also provide three new models for ACCENTOR explicitly trained to predict user goals and to generate contextually relevant chit-chat responses.

Automatic and human evaluations show that, compared with the state of-the-art task-oriented baseline, our models can code-switch between task and chit-chat to be more engaging, interesting, knowledgeable, and humanlike, while maintaining competitive task performance.

For more details, please refer to this [paper][accentor_arxiv].

## Data

* ```v1.0/candidates-{sgd,multiwoz}.json```: Annotated chit-chat candidates. The format is as follows.

```
{
 "dialogue 1 / id": [
 [
  dialogue 1 / candidate 1 / turn id,
  dialogue 1 / candidate 1 / position,
  dialogue 1 / candidate 1 / candidate,
  dialogue 1 / candidate 1 / justification
 ],
 [
  dialogue 1 / candidate 2 / turn id,
  ...
 ],
 ...
 ],
 "dialogue 2 / id": [
 ...
 ],
}
```

* Folder ```v1.0/accentor-sgd```: The augmented SGD dataset. The format follows the original SGD dataset, with two additional keys (i.e., ```beginning``` and ```end```) that store lists of ```(candidate, justification)``` pairs. The folder is generated by ```v1.0/accentor-sgd.py``` (with ```v1.0/candidates-sgd.json``` and the [original SGD dataset](https://github.com/google-research-datasets/dstc8-schema-guided-dialogue) as input). 

* ```v1.0/accentor-multiwoz-1k.json```: 1K augmented MultiWOZ 2.1 dialogues. The format follows the original MultiWOZ dataset, with two additional keys (i.e., ```beginning``` and ```end```) that store lists of ```(candidate, justification)``` pairs. The file is generated by ```v1.0/accentor-multiwoz.py``` (with ```v1.0/candidates-multiwoz.json``` and the [original MultiWOZ 2.1 dataset](https://github.com/budzianowski/multiwoz) as input).

## Citations

If you want to publish experimental results with our datasets or use the baseline models, please cite the following article ([pdf][accentor_arxiv]):
```
@article{sun2020adding,
  title={Adding Chit-Chats to Enhance Task-Oriented Dialogues},
  author={Sun, Kai and Moon, Seungwhan and Crook, Paul and Roller, Stephen and Silvert, Becka and Liu, Bing and Wang, Zhiguang and Liu, Honglei and Cho, Eunjoon and Cardie, Claire},
  journal={arXiv preprint arXiv:2010.12757},
  year={2020}
}
```

## License

ACCENTOR is released under [CC-BY-SA-4.0](https://creativecommons.org/licenses/by-sa/4.0/legalcode), see [LICENSE](LICENSE) for details.

[accentor_arxiv]:https://arxiv.org/abs/2010.12757


