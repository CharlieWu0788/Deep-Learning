"""
Translate with the model you trained at home.
COMP 395 – Deep Learning

Put this file next to seq2seq_translation_wandb.py and translation_model.pt, then:
    python translate.py                       # type sentences, empty line to quit
    python translate.py "We see the sky."     # translate one sentence
"""

import sys

import torch

from seq2seq_translation_wandb import Seq2Seq, translate

saved = torch.load("translation_model.pt", map_location="cpu")
cfg = saved["config"]
model = Seq2Seq(len(saved["src_vocab"]), len(saved["tgt_vocab"]),
                cfg["embed_dim"], cfg["hidden_size"], cfg["cell"])
model.load_state_dict(saved["model"])


def show(sentence):
    print(translate(model, sentence, saved["src_vocab"], saved["tgt_vocab"], cfg["max_len"]))


if len(sys.argv) > 1:
    show(" ".join(sys.argv[1:]))
else:
    while (sentence := input("English> ").strip()):
        show(sentence)
