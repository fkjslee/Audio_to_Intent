import bidict
import yaml
import logging
import os
import torch
import random
from typing import Optional
from torch.utils.data import Dataset
from transformers import BertTokenizer

logger = logging.getLogger(__name__)


class Vocabulary:
    def __init__(self, dir_path, slot_name):
        if slot_name == 'intent':
            with open(dir_path + "/intent_label.yml", "r", encoding="utf-8") as f:
                label_map = yaml.load(f.read(), yaml.FullLoader)
                self.vocab = bidict.bidict({key: idx for idx, key in enumerate(label_map.keys())})
        else:
            with open(dir_path + "/slot_label.yml", "r", encoding="utf-8") as f:
                label_map = yaml.load(f.read(), yaml.FullLoader)
                if slot_name == 'slot':
                    self.vocab = bidict.bidict({key: idx for idx, key in enumerate(label_map.keys())})
                else:
                    self.vocab = bidict.bidict({key: idx for idx, key in enumerate(label_map[slot_name])})

    def stoi(self, word):
        return self.vocab.get(word, self.vocab['UNK'])

    def itos(self, idx):
        return self.vocab.inverse.get(idx, 'UNK')



class WordDataset(Dataset):

    def __init__(self, data, which_dataset: str, config):
        self.which_dataset = which_dataset
        self.config = config
        try:
            self.sentence_list, self.intent_list, self.slot_list = data
        except KeyError:
            assert False, logging.error("key error config={}".format(str(config)))
        logger.info(str(self))

    def __len__(self):
        return len(self.intent_list)

    # input_ids, attention_mas, token_type_ids, intent_label_id, slot_labels_ids
    def __getitem__(self, idx):
        return WordDataset.generate_feature_and_label(self.sentence_list[idx], self.which_dataset, self.config, self.slot_list[idx], self.intent_list[idx], self.which_dataset == 'slot')

    def __str__(self):
        res = '\nLoad {} dataset Complete\nDataset total length = {}\n'.format(self.which_dataset, len(self))
        show_sample_num = min(len(self), 3)
        res += 'Randomly show {} samples\n'.format(show_sample_num)
        for i in range(show_sample_num):
            idx = random.randint(0, len(self)-1)
            res += "Sample {}:\n".format(i)
            res += "Sentences: {}\n".format(self.sentence_list[idx])
            res += "Intent: {}\n".format(self.intent_list[idx])
            res += "Slot: {}\n".format(self.slot_list[idx])
            res += "\n"
        return res

    @staticmethod
    def generate_feature_and_label(sentence: list, which_dataset, config, slot_list: Optional[list] = None,
                                   label: Optional[str] = None, pred_slot=False):
        """
        generate instance of sentence(feature), slot and intent(label) after been ont-hot encoded
        :param sentence: sentence which need to be converted, which should be tokenizer first.
        :param slot_list: sentence's slot label.
        :param label: label of the sentence.
        :return: one_hot encoding result. example
        example:
            generate_feature_and_label(['Move', 'Tsinghua University', 'to', 'Guangdong'],
                                        ['O', 'B-moved_object', 'O', 'B-moved_position'],
                                        "moved_object"
                                       )
            (Tsinghua University may be tokenized to two word), add CLS in front and SEP in tail
            So, sentence may be understood as ['CLS', 'Move', 'Tsinghua', 'University', 'to', 'GUANGDONG', 'SEP']
            return:
                ([101, 123, 124, 125, 126, 127, 102, 0, 0, 0, 0, 0, ...., 0],
                [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, ..., 0],
                [0, 0, 0, ...., 0],
                2,
                [-2, 0, 1, -2, 1, 2, -2, ]
            (-2: ignore slot, -1: unknow slot)
        """
        if slot_list is None:
            slot_list = [None] * len(sentence)
        input_ids, attention_mask, slot_label_ids = WordDataset.one_hot_encoding_sentence(sentence, slot_list, config)
        if which_dataset == 'intent':
            label_id = -1 if label is None else config['vocab'][which_dataset].stoi('UNK')
        elif which_dataset == 'slot':
            label_id = slot_label_ids
        else:
            label_id = config['vocab'][which_dataset].stoi('UNK')
            for slot_str, slot_type in zip(sentence, slot_list):
                if slot_type == which_dataset:
                    label_id = config['vocab'][which_dataset].stoi(slot_str)
        return list(map(lambda x: torch.tensor(x, dtype=torch.int64), [input_ids, attention_mask, label_id]))

    @staticmethod
    def one_hot_encoding_sentence(sentence: list, slot_list: list, config):
        """
        one-hot encoding sentence and slot by tokenizer
        param example can be seen in generate_feature_and_label
        """
        tokens = []
        slot_label_ids = []
        for word, slot_label in zip(sentence, slot_list):
            word_tokens = config["tokenizer"].tokenize(word)
            if not word_tokens:
                word_tokens = config["tokenizer"].unk_token
            tokens.extend(word_tokens)
            slot_vocab = config['vocab']['slot']
            slot_label_ids.extend(
                [-1 if slot_label is None else slot_vocab.stoi(slot_label)] + [-2] * (len(word_tokens) - 1))

        input_id = config["tokenizer"].convert_tokens_to_ids(tokens)
        if len(tokens) > config["feature_length"] - 2:
            attention_mask = [1] * config.feature_length
            slot_label_ids = [-2] + slot_label_ids[0: WordDataset.feature_length - 2] + [-2]
            input_id = [config["tokenizer"].cls_token_id] + input_id[0: config["feature_length"] - 2] + [
                config["tokenizer"].sep_token_id]
        else:
            add_len = config["feature_length"] - 2 - len(input_id)
            attention_mask = [1] * (len(input_id) + 2) + [0] * add_len
            slot_label_ids = [-2] + slot_label_ids + [-2] + [-2] * add_len
            input_id = [config["tokenizer"].cls_token_id] + input_id + [config["tokenizer"].sep_token_id] + [0] * add_len
        return input_id, attention_mask, slot_label_ids
