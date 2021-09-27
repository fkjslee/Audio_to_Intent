import os
import copy
import json
import logging
import yaml
import torch
from torch.utils.data import TensorDataset
from preprocess.dataaugment import augmentTrainData

from utils import get_intent_labels, get_slot_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example. [train, valid, test]-<id of instance> example: train-5
        words: list. The words of the sequence. example: ['Book', 'a', 'ticket']
        intent_label: (Optional) int. The intent of sentences. example: 2
        slot_labels: (Optional) list of int. The slot labels of the example. example: [3, 2, 1]
    """

    def __init__(self, guid, words, intent_label=None, slot_labels=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.slot_labels = slot_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """
        A single set of features of data.
        input_ids: sentence's word to id, means word's id in corpus, example: [101, 3123, 1092, 0, 0, 0, 0]
        attention_mask: which id is valid, example: [1, 1, 1, 0, 0, 0, 0]
        token_type_ids: explain what type of each token
    """
    def __init__(self, input_ids: list, attention_mask: list, token_type_ids: list, intent_label_id: int, slot_labels_ids: list):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.slot_labels_ids = slot_labels_ids

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)

    def _create_examples(self, texts, intents=None, slots=None, set_type='train'):
        """
        Creates examples for the training and valid sets.
        one-hot encoding for word intent slot.
        """
        examples = []
        if intents is None:
            intents = [None] * len(texts)
            slots = [None] * len(texts)
        for i, (text, intent, slot) in enumerate(zip(texts, intents, slots)):
            guid = "%s-%s" % (set_type, i)
            words = text.split()
            intent_label = None if intent is None else self.intent_labels.index(
                intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
            slot_labels = []
            if slot is None:
                slot_labels = [None] * len(words)
            else:
                for s in slot.split():
                    slot_labels.append(
                        self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))

            assert len(words) == len(slot_labels)
            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, slot_labels=slot_labels))
        return examples

    def get_examples(self, mode):
        """
        Args:
            mode: train, valid, test
        """
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)
        logger.debug("LOOKING AT {}".format(data_path))
        f = open(os.path.join(data_path, "labeled_sentences.yml"), 'r', encoding='utf-8')
        d = yaml.load(f.read(), yaml.FullLoader)
        texts = []
        intents = []
        slots = []
        for key in d:
            texts.append(key['sentence'])
            intents.append(key['intent'])
            slots.append(key['slot'])
        if mode == "train":
            texts, intents, slots = augmentTrainData(texts, intents, slots)
        return self._create_examples(texts=texts, intents=intents, slots=slots, set_type=mode)  # one-hot encoding text, intent, slot


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True) -> list:
    """
    convert features to one-hot encoding by tokenizer
    Args:
        examples: main object to be converted
        max_seq_len: how long of ids. For every sentence share same feature columns
        tokenizer: convert word to id
        pad_token_label_id: remain word's id, for example, microsoft company's id is 555, microsoft company may be converted to [555, -1] if pad_token_label_id=-1
        cls_token_segment_id: special word CLS's id
        pad_token_segment_id: special word PAD's id
        sequence_a_segment_id: valid token's mask
        mask_padding_with_zero:
        return: a list of feature
    """
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    logger = logging.getLogger(__name__)
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.debug("Writing example %d of %d" % (ex_index, len(examples)))

        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real sentences.txt id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([-1 if slot_label is None else int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)
        assert len(
            slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(
            len(slot_labels_ids), max_seq_len)

        intent_label_id = -1 if example.intent_label is None else int(example.intent_label)

        if ex_index < 5:
            logger.debug("*** Example ***")
            logger.debug("guid: %s" % example.guid)
            logger.debug("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.debug("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.debug("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.debug("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.debug(
                "intent_label: %s (id = %s)" % (example.intent_label, intent_label_id))
            logger.debug("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          slot_labels_ids=slot_labels_ids
                          ))

    return features


def load_and_cache_examples(args, tokenizer, mode) -> TensorDataset:
    """
    generate dataset, if has cache, load cache.
    Args:
        args: commandline args.
        tokenizer: a tool convert sentences to word.
        mode: in ['train', 'valid', 'test']
    """
    processor = JointProcessor(args)

    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            args.task,
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if os.path.exists(cached_features_file):
        logger.debug("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.debug("Creating features from dataset file at %s", os.path.join(args.data_dir, args.task))
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "valid":
            examples = processor.get_examples("valid")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, valid, test is available")

        # Use cross entropy ignore index as padding sentences.txt id so that only real sentences.txt ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.debug("Saving features into cached file %s", cached_features_file)
        # torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
    return dataset
