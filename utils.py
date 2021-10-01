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


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--log_level", default="info", type=str, help="The name of the task to train")
    parser.add_argument("--task", default="qiyuan", type=str, help="The name of the task to train")
    parser.add_argument("--data_dir", default="data", type=str, help="The input data dir")

    parser.add_argument('--seed', type=int, default=1234, help="random seed for initialization")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int, help="Batch size for evaluation.")
    parser.add_argument("--max_seq_len", default=50, type=int,
                        help="The maximum total input sequence length after tokenization.")
    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")


    parser.add_argument("--do_load", action="store_true", help="Whether to load model.")
    parser.add_argument("--do_valid", action="store_true", help="Whether to validate model.")
    parser.add_argument("--device", default="cuda:0", help="Run device, default is cuda:0")

    parser.add_argument("--ignore_index", default=0, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')
    parser.add_argument('--command_server_addr', default=None, help='Address of server which receive intent')
    parser.add_argument('--command_server_port', type=int, default=None, help='Port of Server which receive intent')

    # CRF option
    parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str,
                        help="Pad token for slot sentences.txt pad (to be ignore when calculate loss)")

    args = parser.parse_args()
    args.model_dir = args.task + "_model"

    if args.task in ["qiyuan"]:
        args.model_name_or_path = 'bert-base-chinese'
    elif args.task in ["atis", "snip"]:
        args.model_name_or_path = 'bert-base-uncased'
    else:
        raise NotImplementedError("Not implement")
    return args
