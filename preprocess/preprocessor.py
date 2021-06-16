import jieba
import yaml
import os
from utils import get_args


def add_jieba_word(d):
    if d is None:
        return
    if isinstance(d, dict):
        for key in d.keys():
            jieba.add_word(key)
            add_jieba_word(d[key])
    else:
        for val in d:
            jieba.add_word(val)


def init_jieba():
    args = get_args()
    f = open(os.path.join(args.data_dir, args.task, "slot_label.yml"), 'r', encoding='utf-8')
    d = yaml.load(f.read(), yaml.FullLoader)
    add_jieba_word(d)


def init_asr():
    f = open(os.path.join("asr", "asrConfig.yml"), 'r', encoding='utf-8')
    d = yaml.load(f.read(), yaml.FullLoader)
    return d
