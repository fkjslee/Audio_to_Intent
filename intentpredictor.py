# reference: https://github.com/monologg/JointBERT
import argparse

from trainer import Trainer
from utils import init_logger, load_tokenizer, read_prediction_text, set_seed, MODEL_CLASSES, MODEL_PATH_MAP, get_args
from data_loader import load_and_cache_examples
from transformers import BertTokenizer
from utils import get_intent_labels, get_slot_labels
import jieba


class IntentPredictor:
    def __init__(self):
        init_logger()
        args = get_args()
        set_seed(args)
        self.tokenizer = load_tokenizer(args)

        train_dataset = load_and_cache_examples(args, self.tokenizer, mode="train")
        dev_dataset = load_and_cache_examples(args, self.tokenizer, mode="dev")
        test_dataset = load_and_cache_examples(args, self.tokenizer, mode="test")

        self.trainer = Trainer(args, train_dataset, dev_dataset, test_dataset)
        self.trainer.train()


    def predict(self, text):
        text = " ".join(jieba.lcut(text))
        print('start to predict', text)
        intent_preds, slot_preds_list = self.trainer.predict([text], self.tokenizer)
        return intent_preds, slot_preds_list
