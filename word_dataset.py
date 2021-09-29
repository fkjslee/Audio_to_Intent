from torch.utils.data import Dataset
from transformers import BertTokenizer
import bidict
import yaml
import logging
import os
import torch

logger = logging.getLogger(__name__)


class WordDataset(Dataset):
    def __init__(self, sentence_list: list, slot_list: list, which_slot: str, intent_list: list, config: map):
        self.sentence_list = sentence_list
        self.slot_list = slot_list
        self.which_slot = which_slot
        self.intent_list = intent_list
        try:
            self.feature_length = config['word_length']
            self.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_name_or_path'])
            self.entity_bidict = WordDataset.build_entity_bidict(config['intent_label_file_path'])
            self.all_slot_bidict = WordDataset.build_slot_bidict(config['slot_label_file_path'])
        except KeyError:
            assert False, logging.error("key error config={}".format(str(config)))
        assert which_slot in self.all_slot_bidict.keys(), "Which_slot must in {}".format(str(self.all_slot_bidict.keys()))
        logger.info(str(self))

    def __len__(self):
        return len(self.intent_list)

    # input_ids, attention_mas, token_type_ids, intent_label_id, slot_labels_ids
    def __getitem__(self, idx):
        input_ids, attention_mask, slot_label_ids = self.one_hot_encoding_sentence(self.sentence_list[idx], self.slot_list[idx])
        return list(map(lambda x: torch.tensor(x, dtype=torch.int64), [input_ids, attention_mask, [0] * len(input_ids), self.entity_bidict[self.intent_list[idx]], slot_label_ids]))

    def __str__(self):
        res = 'Load dataset Complete\nDataset total length = {}\n'.format(len(self))
        show_sample_num = min(len(self), 5)
        res += 'Show {} samples\n'.format(show_sample_num)
        for i in range(show_sample_num):
            res += '{}\n'.format(str(self.__getitem__(i)))
        return res

    def get_entity_bidict(self):
        return self.entity_bidict

    def get_all_slot_bidict(self):
        return self.all_slot_bidict

    @staticmethod
    def build_entity_bidict(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            label_map = yaml.load(f.read(), yaml.FullLoader)
            label_bidict = bidict.bidict()
            for label in set(label_map.keys()):
                label_bidict[label] = len(label_bidict)
        return label_bidict

    @staticmethod
    def build_slot_bidict(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            label_map = yaml.load(f.read(), yaml.FullLoader)
            all_slot_bidict = {}
            for slot_key in label_map.keys():
                slot_bidict = bidict.bidict()
                if isinstance(label_map[slot_key], list):
                    for instance in label_map[slot_key]:
                        slot_bidict[instance] = len(slot_bidict)
                all_slot_bidict[slot_key] = slot_bidict
        return all_slot_bidict


    def one_hot_encoding_sentence(self, sentence: list, slot_list: list):
        """
        convert sentence to one_hot encoding by tokenizer
        example:
        :param sentence: sentence which need to be converted, which should be tokenizer first. example: ['Move', 'Tsinghua University', 'to', 'Guangdong']
        :return: one_hot encoding result
        """
        tokens = []
        slot_label_ids = []
        for word, slot_label in zip(sentence, slot_list):
            word_tokens = self.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = self.tokenizer.unk_token
            tokens.extend(word_tokens)
            slot_label_ids.extend([list(self.all_slot_bidict.keys()).index(slot_label)] + [0] * (len(word_tokens) - 1))

        input_id = self.tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) > self.feature_length - 2:
            attention_mask = [1] * self.feature_length
            slot_label_ids = slot_label_ids[0: self.feature_length]
            input_id = [self.tokenizer.cls_token_id] + input_id[0: self.feature_length - 2] + [self.tokenizer.sep_token_id]
        else:
            attention_mask = [1] * len(input_id) + [0] * (self.feature_length - len(input_id))
            slot_label_ids += [0] * (self.feature_length - len(slot_label_ids))
            input_id = [self.tokenizer.cls_token_id] + input_id + [self.tokenizer.sep_token_id] + [0] * (self.feature_length - 2 - len(input_id))
        return input_id, attention_mask, slot_label_ids
