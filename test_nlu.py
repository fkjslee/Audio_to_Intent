# reference: https://github.com/monologg/JointBERT
import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP, get_args
from data_loader import load_and_cache_examples
from transformers import BertTokenizer
from utils import get_intent_labels, get_slot_labels


def main(args):
    init_logger()
    set_seed(args)
    tokenizer = load_tokenizer(args)

    train_dataset = load_and_cache_examples(args, tokenizer, mode="train")
    dev_dataset = load_and_cache_examples(args, tokenizer, mode="dev")
    test_dataset = load_and_cache_examples(args, tokenizer, mode="test")

    trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
    trainer.train()

    cmd = input('input cmd')
    while not cmd == "退出":
        intent_preds, slot_preds_list = trainer.predict([cmd], tokenizer)
        print(intent_preds)
        print(slot_preds_list)
        print(get_intent_labels(args))
        print(get_slot_labels(args))
        cmd = input('input cmd')

    print('eval', args.do_eval)

    trainer.evaluate("test")


if __name__ == '__main__':
    args = get_args()
    main(args)
