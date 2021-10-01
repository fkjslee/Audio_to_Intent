import os
import logging
from tqdm import tqdm, trange
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import compute_metrics, get_args
from transformers import BertTokenizer
from model import JointBERT
from data import WordDataset
import yaml

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
            "pretrained_model_name_or_path": "bert-base-chinese",
            "intent_label_file_path": os.path.abspath(os.path.join(args.data_dir, args.task, "intent_label.yml")),
            "slot_label_file_path": os.path.abspath(os.path.join(args.data_dir, args.task, "slot_label.yml")),
        }
        logger.info("Init word dataset with config: {}".format(str(config)))
        WordDataset.init_word_dataset(config)

        if self.args.do_load:
            try:
                self.load_model()
            except Exception:
                logger.critical("Load model failed! % ")
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
                scheduler.step()  # Update learning rate schedule
                global_step += 1
        self.save_model()

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

    def predict_sentence(self, sentence: list):
        instance = WordDataset.generate_feature_and_label(sentence)
        batch = list(map(lambda x: x.unsqueeze(0), instance))
        batch = tuple(t.to(self.device) for t in batch)
        result = self.model(input_ids=batch[0], attention_mask=batch[1], intent_label_ids=batch[3],
                            slot_labels_ids=batch[4], token_type_ids=batch[2])
        return result['intent_logits'][0], result['slot_logits'][0]

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.debug("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first! model path: %s " % str(self.args.model_dir))

        try:
            self.model = JointBERT.from_pretrained(self.args.model_dir,
                                                   args=self.args,
                                                   intent_label_lst=WordDataset.intent_bidict,
                                                   slot_label_lst=WordDataset.all_slot_dict)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
