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


class WordDataset(Dataset):
    tokenizer = None
    each_slot_dict = None
    feature_length = None

    def __init__(self, data, which_slot: str):
        self.which_slot = which_slot
        try:
            self.sentence_list, self.intent_list, self.slot_list = data
        except KeyError:
            assert False, logging.error("key error config={}".format(str(config)))
        assert which_slot in self.each_slot_dict.keys(), "Which_slot must in {}".format(str(self.each_slot_dict.keys()))
        logger.info(str(self))

    def __len__(self):
        return len(self.intent_list)

    # input_ids, attention_mas, token_type_ids, intent_label_id, slot_labels_ids
    def __getitem__(self, idx):
        return WordDataset.generate_feature_and_label(self.sentence_list[idx], self.which_slot, self.slot_list[idx], self.intent_list[idx])

    def __str__(self):
        res = '\nLoad {} dataset Complete\nDataset total length = {}\n'.format(self.which_slot, len(self))
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
    def init_word_dataset(config):
        WordDataset.tokenizer = BertTokenizer.from_pretrained(config['pretrained_model_name_or_path'])
        WordDataset.each_slot_dict = WordDataset.build_slot_bidict(config['slot_label_file_path'], config['intent_label_file_path'])
        WordDataset.feature_length = config['word_length']

    @staticmethod
    def build_intent_bidict(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            label_map = yaml.load(f.read(), yaml.FullLoader)
            return bidict.bidict({key: idx for idx, key in enumerate(label_map.keys())})

    @staticmethod
    def build_slot_bidict(slot_label_path, intent_label_path):
        with open(slot_label_path, "r", encoding="utf-8") as f:
            label_map = yaml.load(f.read(), yaml.FullLoader)
            each_slot_dict = {}
            for slot_key in label_map.keys():
                if isinstance(label_map[slot_key], list):
                    each_slot_dict[slot_key] = bidict.bidict({key: idx for idx, key in enumerate(label_map[slot_key])})
            each_slot_dict['intent'] = WordDataset.build_intent_bidict(intent_label_path)
        return each_slot_dict

    @staticmethod
    def generate_feature_and_label(sentence: list, which_slot, slot_list: Optional[list] = None,
                                   label: Optional[str] = None):
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
        assert WordDataset.each_slot_dict is not None, "not initialize dataset yet!"
        if slot_list is None:
            slot_list = [None] * len(sentence)
        if which_slot == 'intent':
            label_id = -1 if label is None else WordDataset.each_slot_dict['intent'][label]
        else:
            label_id = WordDataset.each_slot_dict[which_slot]['UNK']
            for slot_str, slot_type in zip(sentence, slot_list):
                if slot_type == which_slot:
                    label_id = WordDataset.each_slot_dict[which_slot][slot_str]
        input_ids, attention_mask = WordDataset.one_hot_encoding_sentence(sentence, slot_list)
        return list(map(lambda x: torch.tensor(x, dtype=torch.int64), [input_ids, attention_mask, label_id]))

    @staticmethod
    def one_hot_encoding_sentence(sentence: list, slot_list: list):
        """
        one-hot encoding sentence and slot by tokenizer
        param example can be seen in generate_feature_and_label
        """
        assert WordDataset.each_slot_dict is not None, "not initialize dataset yet!"
        tokens = []
        for word, slot_label in zip(sentence, slot_list):
            word_tokens = WordDataset.tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = WordDataset.tokenizer.unk_token
            tokens.extend(word_tokens)

        input_id = WordDataset.tokenizer.convert_tokens_to_ids(tokens)
        if len(tokens) > WordDataset.feature_length - 2:
            attention_mask = [1] * WordDataset.feature_length
            input_id = [WordDataset.tokenizer.cls_token_id] + input_id[0: WordDataset.feature_length - 2] + [
                WordDataset.tokenizer.sep_token_id]
        else:
            add_len = WordDataset.feature_length - 2 - len(input_id)
            attention_mask = [1] * (len(input_id) + 2) + [0] * add_len
            input_id = [WordDataset.tokenizer.cls_token_id] + input_id + [WordDataset.tokenizer.sep_token_id] + [0] * add_len
        return input_id, attention_mask
