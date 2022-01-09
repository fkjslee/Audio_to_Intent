import sys
import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score
from typing import List

from transformers import BertTokenizer
import argparse
import yaml
import bisect

logger = logging.getLogger(__name__)


def augmentTrainData(texts, intents, slots):
    """
    augment data:
    replace each slot with all examples
    e.g.
    moved_object=['a', 'b']
    sentences = ["move x to y"]
    after augment:
    sentences = ["move x to y", "move a to y", "move b to y"]
    """
    try:
        args = get_args()
        f = open(os.path.join(args.data_dir, args.task, "slot_label.yml"), 'r', encoding='utf-8')
        d = yaml.load(f.read(), yaml.FullLoader)
    except FileNotFoundError:
        assert False, logging.error("augment data failed!, check if data path:{} is correct!".format(os.path.join(args.data_dir, args.task, "slot_label.yml")))

    def dfs(example, dep, augment_text, augment_intent, augment_slot, replace_words, all_texts, all_intents, all_slots):
        if len(example) == dep:
            all_texts.append(" ".join(augment_text))
            all_intents.append(augment_intent)
            all_slots.append(augment_slot)
            return
        if isinstance(replace_words[augment_slot[dep]], list):
            for replace_word in replace_words[augment_slot[dep]]:
                augment_text[dep] = replace_word
                dfs(example, dep+1, augment_text, augment_intent, augment_slot, replace_words, all_texts, all_intents, all_slots)
        else:
            augment_text[dep] = example[dep]
            dfs(example, dep+1, augment_text, augment_intent, augment_slot, replace_words, all_texts, all_intents, all_slots)

    def balance_train_data(texts, intents, slots):
        # sort by intent
        idx = np.argsort(intents)
        texts = [texts[i] for i in idx]
        intents = [intents[i] for i in idx]
        slots = [slots[i] for i in idx]

        # randomly add 100 example every intent
        unique_intents = np.unique(intents)
        original_len = len(intents)
        for intent in unique_intents:
            for _ in range(100):
                random_idx = random.randint(bisect.bisect_left(intents, intent, 0, original_len), bisect.bisect_right(intents, intent, 0, original_len) - 1)
                texts.append(texts[random_idx])
                intents.append(intents[random_idx])
                slots.append(slots[random_idx])
        return texts, intents, slots

    store_texts = []
    store_intents = []
    store_slots = []
    for text, intent, slot in zip(texts, intents, slots):
        text = text.split(" ")
        dfs(text, 0, text.copy(), intent, slot, d, store_texts, store_intents, store_slots)
    return balance_train_data(store_texts, store_intents, store_slots)


def init_logger():
    log_levels = {"debug": logging.DEBUG, "info": logging.INFO, "warning": logging.WARNING, "error": logging.ERROR, "critical": logging.CRITICAL}
    if not os.path.exists("./log"):
        os.makedirs(path)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.DEBUG, filename="./log/log_file.txt")
    console = logging.StreamHandler()
    console.setLevel(log_levels[get_args().log_level])
    console.stream = sys.stdout
    logging.getLogger('').addHandler(console)
    logger.info("args: %s" % str(get_args()))


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device != "cpu" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(slot_preds) and intent_preds.shape == intent_labels.shape and slot_preds.shape == slot_labels.shape
    results = {'intent_acc': (intent_preds == intent_labels).mean(), 'slot_acc': (slot_preds == slot_labels).mean()}
    return results


def get_data_from_path(data_path, augment=True):
    try:
        f = open(os.path.join(data_path, "labeled_sentences.yml"), 'r', encoding='utf-8')
        d = yaml.load(f.read(), yaml.FullLoader)
        space_cut_sentences = []
        intents = []
        slots = []
        for key in d:
            assert len(key['sentence'].split(' ')) == len(key['slot'])
            space_cut_sentences.append(key['sentence'])
            intents.append(key['intent'])
            slots.append(key['slot'])
        if augment:
            space_cut_sentences, intents, slots = augmentTrainData(space_cut_sentences, intents, slots)
        word_list_sentences = [sentence.split(' ') for sentence in space_cut_sentences]
        return word_list_sentences, intents, slots
    except FileNotFoundError:
        assert False, logging.error("load data failed!, check if data path:{} is correct!".format(data_path))


def get_args():
    parser = argparse.ArgumentParser()

    # model config
    parser.add_argument("--log_level", default="info", type=str, help="The name of the task to train")
    parser.add_argument("--task", default="qiyuan", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="data", type=str, help="The input data dir")
    parser.add_argument("--predict_slots", default=[], nargs='+', help="Which slots to predict")
    parser.add_argument("--device", default="cuda:0", help="Run device, default is cuda:0")
    parser.add_argument("--do_load", action="store_true", help="Whether to load model.")
    parser.add_argument("--do_valid", action="store_true", help="Whether to validate model.")
    parser.add_argument("--crf", action="store_true", help="Whether to use crf.")

    # model parameter
    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--train_ratio", default=-1.0, type=float, help="Percentage of split train and valid set. Default 0.8 if need to validation else 1.0")

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # network
    parser.add_argument('--command_server_addr', default=None, help='Address of server which receive intent')
    parser.add_argument('--command_server_port', type=int, default=None, help='Port of Server which receive intent')

    # gui
    parser.add_argument("--gui", action="store_true", help="Whether to visualize audio. (recommended if it could be)")

    args = parser.parse_args()
    args.model_dir = args.task + "_model"

    if args.train_ratio < 0:
        if args.do_valid:
            args.train_ratio = 0.8
        else:
            args.train_ratio = 1.0


    if args.task in ["qiyuan", "VGA"]:
        args.model_name_or_path = 'bert-base-chinese'
        if not args.predict_slots:
            args.predict_slots = ['intent', 'B-moved_object', 'B-moved_position']
    elif args.task in ["atis", "snip"]:
        args.model_name_or_path = 'bert-base-uncased'
    else:
        raise NotImplementedError("Not implement")
    return args
