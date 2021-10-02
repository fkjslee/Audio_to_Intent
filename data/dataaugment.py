from utils import get_args
import os
import yaml

"""
augment train data:
replace each slot with all examples
e.g.
moved_object=['a', 'b']
sentences = ["move x to y"]
after augment:
sentences = ["move x to y", "move a to y", "move b to y"]
"""


def augmentTrainData(texts, intents, slots):
    args = get_args()
    f = open(os.path.join(args.data_dir, args.task, "slot_label.yml"), 'r', encoding='utf-8')
    d = yaml.load(f.read(), yaml.FullLoader)
    store_texts = []
    store_intents = []
    store_slots = []
    been_augment_entity = []
    for text, intent, slot in zip(texts, intents, slots):
        for i, (word, entity) in enumerate(zip(text, slot)):
            if isinstance(d[entity], list) and entity not in been_augment_entity:
                been_augment_entity.append(entity)
                for example in d[entity]:
                    new_text = text.split(" ")
                    new_text[i] = example
                    new_text = " ".join(new_text)
                    store_texts.append(new_text)
                    store_intents.append(intent)
                    store_slots.append(slot)
    texts.extend(store_texts)
    intents.extend(store_intents)
    slots.extend(store_slots)
    return texts, intents, slots
