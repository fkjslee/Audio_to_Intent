import os
import logging
from tqdm import tqdm, trange
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertConfig, AdamW, get_linear_schedule_with_warmup

from utils import compute_metrics, get_intent_labels, get_slot_labels, get_args, load_tokenizer
from data_loader import JointProcessor, convert_examples_to_features
from transformers import BertTokenizer
from data_loader import load_and_cache_examples
from model import JointBERT

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self):
        self.args = get_args()
        args = self.args
        self.tokenizer = load_tokenizer(self.args)

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = BertConfig, JointBERT, BertTokenizer
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)
        self.model = self.model_class.from_pretrained(args.model_name_or_path, config=self.config, args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)
        # GPU or CPU
        self.device = args.device
        self.model.to(self.device)

        if self.args.do_load:
            self.load_model()
        else:
            self.train(load_and_cache_examples(args, self.tokenizer, 'train'))

        if self.args.do_valid:
            self.valid(load_and_cache_examples(args, self.tokenizer, 'valid'))

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
                                                    num_training_steps=len(train_dataloader) * self.args.num_train_epochs)

        global_step = 0

        for _ in range(int(self.args.num_train_epochs)):
            epoch_iterator = tqdm(train_dataloader, desc="Epoch %d in %d" % (_, self.args.num_train_epochs), position=0, file=sys.stdout)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4], 'token_type_ids': batch[2]}
                outputs = self.model(**inputs)
                loss = outputs[0]
                epoch_iterator.set_postfix(loss=loss.item())
                self.model.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                global_step += 1
        self.save_model()

    def valid(self, dataset, mode='valid'):

        valid_sampler = SequentialSampler(dataset)
        valid_dataloader = DataLoader(dataset, sampler=valid_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.debug("***** Running evaluation on %s dataset *****", mode)
        logger.debug("  Num examples = %d", len(dataset))
        logger.debug("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'intent_label_ids': batch[3],
                          'slot_labels_ids': batch[4], 'token_type_ids': batch[2]}
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(),
                                                axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def predict(self, sentences):
        tokenizer = self.tokenizer
        processor = JointProcessor(self.args)
        examples = processor._create_examples(sentences)
        features = convert_examples_to_features(examples, self.args.max_seq_len, tokenizer,
                                                pad_token_label_id=self.args.ignore_index)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
        all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

        batch = [all_input_ids, all_attention_mask, all_token_type_ids, all_intent_label_ids, all_slot_labels_ids]
        batch = tuple(t.to(self.device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'intent_label_ids': batch[3],
                      'slot_labels_ids': batch[4], 'token_type_ids': batch[2]}
            outputs = self.model(**inputs)
            tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
        intent_preds = intent_logits.detach().cpu().numpy()
        intent_preds = np.argmax(intent_preds, axis=1)

        if self.args.use_crf:
            # decode() in `torchcrf` returns list with best index directly
            slot_preds = np.array(self.model.crf.decode(slot_logits))
        else:
            slot_preds = slot_logits.detach().cpu().numpy()

        slot_preds = np.argmax(slot_preds, axis=2)

        out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]

        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
        return intent_preds, slot_preds_list

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
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")
