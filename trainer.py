import os
import logging
import torch
import yaml
import sys
import numpy as np

from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup
from utils import compute_metrics, get_args
from transformers import BertTokenizer
from model import JointBERT
from data import WordDataset
from typing import Tuple, List

logger = logging.getLogger(__name__)


class Trainer(object):
    r"""
    trainer for intent and slot recognizing
    """

    def __init__(self):
        self.args = get_args()
        args = self.args
        self.config = BertConfig.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.device = args.device

        config = {
            "word_length": 50,
            "pretrained_model_name_or_path": args.model_name_or_path,
            "intent_label_file_path": os.path.abspath(os.path.join(args.data_dir, args.task, "intent_label.yml")),
            "slot_label_file_path": os.path.abspath(os.path.join(args.data_dir, args.task, "slot_label.yml")),
        }
        logger.info("Init word dataset with config: {}".format(str(config)))
        WordDataset.init_word_dataset(config)

        if self.args.do_load:
            self.load_model()
        else:
            self.model = JointBERT.from_pretrained(args.model_name_or_path, config=self.config, args=args,
                                                   intent_label_lst=WordDataset.intent_bidict,
                                                   slot_label_lst=WordDataset.all_slot_dict)
            self.model.to(self.device)

            self.train(WordDataset(os.path.join(args.data_dir, args.task, "train"), "B-moved_object", "train"))

        if self.args.do_valid:
            self.valid(WordDataset(os.path.join(args.data_dir, args.task, "valid"), "B-moved_object", "valid"))


    def train(self, train_dataset):
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=len(
                                                        train_dataloader) * self.args.num_train_epochs)

        global_step = 0

        for _ in range(int(self.args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc="Epoch %d in %d" % (_, self.args.num_train_epochs), position=0,
                                  file=sys.stdout)
            for batch in epoch_iterator:
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                result = self.model(input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[3],
                                    slot_labels_ids=batch[4], token_type_ids=batch[2])
                loss = result['total_loss']
                epoch_iterator.set_postfix(loss=loss.item())
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                global_step += 1
        self.save_model()
        logger.info("***** Train Model Complete with train last batch's total loss = {} *****".format(str(loss.item())))

    def valid(self, dataset, mode='valid'):

        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.debug("  Batch size = %d", self.args.eval_batch_size)

        valid_sampler = SequentialSampler(dataset)
        valid_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=self.args.eval_batch_size)

        eval_loss = 0.0
        self.model.eval()
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                result = self.model(input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[3],
                                    slot_labels_ids=batch[4], token_type_ids=batch[2])
                tmp_eval_loss = result['total_loss']

            eval_loss += tmp_eval_loss.mean().item()

        eval_loss = eval_loss / len(valid_dataloader)
        results = {
            "loss": eval_loss
        }

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def predict_sentence(self, space_cut_sentence: str) -> Tuple[list, list, List[list], List[list]]:
        """
        predict a sentence's intent and slot
        :param space_cut_sentence: sentence which split by space
        :return: intent_str and intent_logit means intent type and its probability, slot type and its probability.
        example
            predict_sentence("Move Tsinghua University to Guangdong")
            return ["Move", "Query"], [0.9, 0.1], [["UNK", "B-moved_object", ...], ["B-moved_object", "UNK", ....], ...], [[0.8, 0.1, ...], [0.8, 0.1, ...], ...]
        """
        sentence = space_cut_sentence.split(' ')
        instance = WordDataset.generate_feature_and_label(sentence)
        batch = list(map(lambda x: x.unsqueeze(0), instance))
        batch = tuple(t.to(self.device) for t in batch)
        result = self.model(input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[3],
                            slot_labels_ids=batch[4], token_type_ids=batch[2])
        intent_logit = result['intent_logits'][0].softmax(dim=0)
        intent_dict = WordDataset.intent_bidict.inverse
        sorted_idx = intent_logit.argsort(descending=True)
        intent_logit = intent_logit[sorted_idx].tolist()
        intent_str = [intent_dict[idx.item()] for idx in sorted_idx]

        slot_logit = result['slot_logits'][0].softmax(dim=1)

        all_word_logit = []
        all_word_str = []
        for word_logit in slot_logit:
            sorted_idx = word_logit.argsort(descending=True)
            all_word_logit.append(word_logit[sorted_idx].tolist())
            slot_dict = WordDataset.all_slot_dict.inverse
            all_word_str.append([slot_dict[idx.item()] for idx in sorted_idx])

        return intent_str, intent_logit, all_word_str, all_word_logit

    def save_model(self):
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.debug("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first! model path: %s " % str(self.args.model_dir))
        try:
            self.model = JointBERT.from_pretrained(self.args.model_dir, args=self.args,
                                                   intent_label_lst=WordDataset.intent_bidict,
                                                   slot_label_lst=WordDataset.all_slot_dict)
            self.model.to(self.device)
            logger.info("***** Load Model Complete *****")
        except Exception:
            raise Exception("Some model files might be missing...")
