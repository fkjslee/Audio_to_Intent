import sys
import os
import random
import logging

import torch
import numpy as np
from seqeval.metrics import precision_score, recall_score, f1_score

from transformers import BertTokenizer
import argparse
import yaml

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
        store_texts = []
        store_intents = []
        store_slots = []
        been_augment_entity = []
        for text, intent, slot in zip(texts, intents, slots):
            for i, (word, entity) in enumerate(zip(text, slot)):
                if isinstance(d[entity], list) and entity not in been_augment_entity:
                    been_augment_entity.append(entity)
                    for example in d[entity]:
                        new_text = text.split(" ")
                        new_text[i] = example
                        new_text = " ".join(new_text)
                        store_texts.append(new_text)
                        store_intents.append(intent)
                        store_slots.append(slot)
        texts.extend(store_texts)
        intents.extend(store_intents)
        slots.extend(store_slots)
        return texts, intents, slots
    except FileNotFoundError:
        assert False, logging.error("augment data failed!, check if data path:{} is correct!".format(os.path.join(args.data_dir, args.task, "slot_label.yml")))


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


def get_data_from_path(data_path):
    try:
        f = open(os.path.join(data_path, "labeled_sentences.yml"), 'r', encoding='utf-8')
        d = yaml.load(f.read(), yaml.FullLoader)
        space_cut_sentences = []
        intents = []
        slots = []
        for key in d:
            space_cut_sentences.append(key['sentence'])
            intents.append(key['intent'])
            slots.append(key['slot'])
        space_cut_sentences, intents, slots = augmentTrainData(space_cut_sentences, intents, slots)
        word_list_sentences = [sentence.split(' ') for sentence in space_cut_sentences]
        return word_list_sentences, intents, slots
    except FileNotFoundError:
        assert False, logging.error("load data failed!, check if data path:{} is correct!".format(data_path))


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_level", default="info", type=str, help="The name of the task to train")
    parser.add_argument("--task", default="qiyuan", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="data", type=str, help="The input data dir")

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=32, type=int, help="Batch size for evaluation.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")


    parser.add_argument("--do_load", action="store_true", help="Whether to load model.")
    parser.add_argument("--do_valid", action="store_true", help="Whether to validate model.")
    parser.add_argument("--device", default="cuda:0", help="Run device, default is cuda:0")

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')
    parser.add_argument('--command_server_addr', default=None, help='Address of server which receive intent')
    parser.add_argument('--command_server_port', type=int, default=None, help='Port of Server which receive intent')

    args = parser.parse_args()
    args.model_dir = args.task + "_model"

    if args.task in ["qiyuan"]:
        args.model_name_or_path = 'bert-base-chinese'
    elif args.task in ["atis", "snip"]:
        args.model_name_or_path = 'bert-base-uncased'
    else:
        raise NotImplementedError("Not implement")
    return args
