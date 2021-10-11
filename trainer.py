import os
import logging
import torch
import yaml
import sys
import numpy as np
import random

from tqdm import tqdm, trange
from utils import get_data_from_path
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
        self.bert_config = BertConfig.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.dataset_config = {
            "word_length": 50,
            "pretrained_model_name_or_path": args.model_name_or_path,
            "intent_label_file_path": os.path.abspath(os.path.join(args.data_dir, args.task, "intent_label.yml")),
            "slot_label_file_path": os.path.abspath(os.path.join(args.data_dir, args.task, "slot_label.yml")),
        }
        self.device = args.device

        self.all_model_name = self.args.predict_slots
        logger.info("Init word dataset with config: {}".format(str(self.dataset_config)))
        WordDataset.init_word_dataset(self.dataset_config)

        if self.args.do_load:
            self.all_models = self.load_model()
        else:
            self.all_models, self.all_dataset = self.generate_model_and_dataset(self.all_model_name)
            for which_slot, model, dataset in zip(self.all_model_name, self.all_models, self.all_dataset):
                self.train(model, dataset[0])
                if self.args.do_valid:
                    self.valid(which_slot, model, dataset[1])
            self.save_model()

    def generate_model_and_dataset(self, all_network_name):
        def generate_model(all_network_name):
            all_models = []
            for which_slot in all_network_name:
                model = JointBERT.from_pretrained(self.args.model_name_or_path, config=self.bert_config, args=self.args,
                                                  intent_label_lst=WordDataset.each_slot_dict[which_slot])
                all_models.append(model.to(self.device))
            return all_models

        def generate_dataset(all_network_name):
            train_ratio = get_args().train_ratio
            assert 0 <= train_ratio <= 1.0
            all_data = get_data_from_path(os.path.join(self.args.data_dir, self.args.task, "train"), augment=True)  # sentences, intents, slots
            idx = [i for i in range(len(all_data[0]))]
            random.shuffle(idx)
            train_idx = idx[:int(len(idx) * train_ratio)]
            valid_idx = idx[int(len(idx) * train_ratio):]
            train_data = []
            valid_data = []
            for j in range(len(all_data)):
                train_data.append([all_data[j][u] for u in train_idx])
                valid_data.append([all_data[j][u] for u in valid_idx])
            all_dataset = []
            for which_slot in all_network_name:
                all_dataset.append((WordDataset(train_data, which_slot), WordDataset(valid_data, which_slot)))
            return all_dataset

        all_model = generate_model(all_network_name)
        all_dataset = generate_dataset(all_network_name)
        return all_model, all_dataset

    def train(self, model, train_dataset):
        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps,
                                                    num_training_steps=len(
                                                        train_dataloader) * self.args.num_train_epochs)

        for _ in range(int(self.args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc="Epoch %d in %d" % (_, self.args.num_train_epochs), position=0,
                                  file=sys.stdout)
            for batch in epoch_iterator:
                model.train()
                batch = tuple(t.to(self.device) for t in batch)
                result = model(input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[2])
                loss = result['total_loss']
                epoch_iterator.set_postfix(loss=loss.item())
                model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()
        logger.info("***** Train Model Complete with train last batch's total loss = {} *****".format(str(loss.item())))

    def valid(self, which_slot, model, dataset):

        logger.info("***** Running evaluation {} *****".format(which_slot))

        valid_sampler = SequentialSampler(dataset)
        valid_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=self.args.eval_batch_size)

        eval_loss = 0.0
        model.eval()
        eval_acc = torch.tensor([]).to(self.device)
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                result = model(input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[2])
                tmp_eval_loss = result['total_loss']

            eval_loss += tmp_eval_loss.mean().item()
            eval_acc = torch.cat((eval_acc, result['intent_acc'].detach().reshape(1)), dim=0)

        eval_loss = eval_loss / len(valid_dataloader)
        results = {
            "loss": eval_loss,
            "acc": eval_acc.mean().item()
        }

        logger.info("***** Eval {} results *****".format(which_slot))
        for key in sorted(results.keys()):
            logger.info("{} = {}".format(key, str(results[key])))

        return results

    def predict_sentence(self, space_cut_sentence: str, which_slot: str) -> Tuple[list, list]:
        """
        predict a sentence's type
        :param space_cut_sentence: sentence which split by space
        :param which_slot: what kind of network you want, 'B-moved_object'? 'B-move_position'?
        :return: intent_str and intent_logit means intent type and its probability, slot type and its probability.
        example
            predict_sentence("Move Tsinghua University to Guangdong", 'B-moved_object')
            return ["Tsinghua University", "Peking University"], [0.9, 0.1]
        """
        model_idx = self.all_model_name.index(which_slot)
        sentence = space_cut_sentence.split(' ')
        instance = WordDataset.generate_feature_and_label(sentence, which_slot)
        batch = list(map(lambda x: x.unsqueeze(0), instance))
        batch = tuple(t.to(self.device) for t in batch)
        result = self.all_models[model_idx](input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[2])
        intent_logit = result['intent_logits'][0].softmax(dim=0)
        intent_dict = WordDataset.each_slot_dict[which_slot].inverse
        sorted_idx = intent_logit.argsort(descending=True)
        intent_logit = intent_logit[sorted_idx].tolist()
        intent_str = [intent_dict[idx.item()] for idx in sorted_idx]

        return intent_str, intent_logit

    def save_model(self):
        for which_slot, model in zip(self.all_model_name, self.all_models):
            path = os.path.join(self.args.model_dir, which_slot)
            if not os.path.exists(path):
                os.makedirs(path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(path)

            torch.save(self.args, os.path.join(path, 'training_args.bin'))
            logger.debug("Saving {} model checkpoint to {}".format(which_slot, path))

    def load_model(self):
        all_models = []
        for which_slot in self.all_model_name:
            path = os.path.join(self.args.model_dir, which_slot)
            if not os.path.exists(path):
                raise Exception("Model doesn't exists! Train first! model path: %s " % str(path))
            try:
                model = JointBERT.from_pretrained(path, args=self.args,
                                                  intent_label_lst=WordDataset.each_slot_dict[which_slot])
                model.to(self.device)
                logger.info("***** Load {} Model Complete *****".format(which_slot))
                all_models.append(model)
            except Exception:
                raise Exception("Some model files might be missing...")
        return all_models
