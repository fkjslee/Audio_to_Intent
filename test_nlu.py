# reference: https://github.com/monologg/JointBERT
import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, get_args
from transformers import BertTokenizer
from utils import get_intent_labels, get_slot_labels


def main(args):
    init_logger()
    set_seed(args)
    Trainer()


if __name__ == '__main__':
    args = get_args()
    main(args)
