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
        with open(os.path.join(data_path, "intent.txt"), encoding="utf-8") as f_intent, \
                open(os.path.join(data_path, "sentence.txt"), encoding="utf-8") as f_s, \
                open(os.path.join(data_path, "slot.txt"), encoding="utf-8") as f_slot:
            intents = f_intent.readlines()
            word_list_sentences = f_s.readlines()
            slots = f_slot.readlines()
            for i in range(len(intents)):
                intents[i] = intents[i].strip()
                word_list_sentences[i] = word_list_sentences[i].strip().split(" ")
                slots[i] = slots[i].strip().split(" ")
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
    parser.add_argument("--manual_input", action="store_true", help="Whether to manual input.")

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
