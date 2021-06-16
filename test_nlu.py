# reference: https://github.com/monologg/JointBERT
import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, get_args
from data_loader import load_and_cache_examples
from transformers import BertTokenizer
from utils import get_intent_labels, get_slot_labels


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    valid_dataset = load_and_cache_examples(args, tokenizer, mode="valid")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, valid_dataset, test_dataset)
    trainer.train()
    if args.do_valid:
        trainer.evaluate("test")


if __name__ == '__main__':
    args = get_args()
    main(args)
