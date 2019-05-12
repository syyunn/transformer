# -*- coding: utf-8 -*-
#/usr/bin/python3
'''
Feb. 2019 by kyubyong park.
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer.

Preprocess the iwslt 2016 datasets.
'''

import os
import errno
import multiprocessing as mp
from tqdm import tqdm
import sentencepiece as spm
from hparams import Hparams
import logging

logging.basicConfig(level=logging.INFO)


def prepro(hp):
    """Load raw data -> Preprocessing -> Segmenting with sentencepice
    hp: hyperparams. argparse.
    """
    logging.info("# Check if raw files exist")

    train1 = "/home/zachary/hdd/nlp/sumdata/train/train.article.txt"
    train2 = "/home/zachary/hdd/nlp/sumdata/train/train.title.txt"
    eval1 = "/home/zachary/hdd/nlp/sumdata/train/valid.article.filter.txt"
    eval2 = "/home/zachary/hdd/nlp/sumdata/train/valid.title.filter.txt"

    for f in (train1, train2, eval1, eval2):
        if not os.path.isfile(f):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), f)

    logging.info("# Preprocessing")
    # train
    _prepro = lambda x:  [line.strip()
                          for line in open(x, 'r').read().split("\n")]

    prepro_train1, prepro_train2, prepro_eval1, prepro_eval2 = \
        _prepro(train1), _prepro(train2), _prepro(eval1), _prepro(eval2)
    print(len(prepro_train1))
    print(len(prepro_train2))
    print(len(prepro_eval1))
    print(len(prepro_eval2))
    assert len(prepro_train1) == len(prepro_train2), "Check if train source and target files match."
    assert len(prepro_eval1) == len(prepro_eval2), "Check if eval train source and eval target files match."

    logging.info("Let's see how preprocessed data look like")
    logging.info("prepro_train1: {}".format(prepro_train1[0]))
    logging.info("prepro_train2: {}".format(prepro_train2[0]))
    logging.info("prepro_eval1: {}".format(prepro_eval1[0]))
    logging.info("prepro_eval2: {}".format(prepro_eval2[0]))

    logging.info("# write preprocessed files to disk")
    os.makedirs("gigaword/prepro", exist_ok=True)

    def _write(sents, fname):
        with open(fname, 'w') as fout:
            fout.write("\n".join(sents))

    _write(prepro_train1, "gigaword/prepro/train.article")
    _write(prepro_train2, "gigaword/prepro/train.title")
    _write(prepro_train1+prepro_train2, "gigaword/prepro/train")  # this is to train sentencepiece
    _write(prepro_eval1, "gigaword/prepro/eval.article")
    _write(prepro_eval2, "gigaword/prepro/eval.title")

    logging.info("# Train a joint BPE model with sentencepiece")
    os.makedirs("gigaword/segmented", exist_ok=True)
    train = '--input=gigaword/prepro/train --pad_id=0 --unk_id=1 \
             --bos_id=2 --eos_id=3\
             --model_prefix=gigaword/segmented/bpe --vocab_size={} \
             --model_type=bpe'.format(hp.vocab_size)
    spm.SentencePieceTrainer.Train(train)

    logging.info("# Load trained bpe model")
    sp = spm.SentencePieceProcessor()
    sp.Load("gigaword/segmented/bpe.model")

    logging.info("# Segment")

    def _segment_and_write(sents, fname):
        with open(fname, "w") as fout:
            for sent in tqdm(sents):
                pieces = sp.EncodeAsPieces(sent)
                fout.write(" ".join(pieces) + "\n")

    _segment_and_write(prepro_train1, "gigaword/segmented/train.article.bpe")
    _segment_and_write(prepro_train2, "gigaword/segmented/train.title.bpe")
    _segment_and_write(prepro_eval1, "gigaword/segmented/eval.article.bpe")
    _segment_and_write(prepro_eval2, "gigaword/segmented/eval.title.bpe")

    logging.info("Let's see how segmented data look like")
    print("train1:", open("gigaword/segmented/train.article.bpe", 'r').readline())
    print("train2:", open("gigaword/segmented/train.title.bpe", 'r').readline())
    print("eval1:", open("gigaword/segmented/eval.article.bpe", 'r').readline())
    print("eval2:", open("gigaword/segmented/eval.title.bpe", 'r').readline())


if __name__ == '__main__':
    hparams = Hparams()
    parser = hparams.parser
    hp = parser.parse_args()
    prepro(hp)
    logging.info("Done")
