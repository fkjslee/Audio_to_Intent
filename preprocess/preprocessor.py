import jieba
import yaml
from utils import get_args
import os


def add_jieba_word(d):
    for key in d.keys():
        jieba.add_word(key)
        if not d[key] is None:
            add_jieba_word(d[key])


def init_jieba():
    args = get_args()
    f = open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')
    d = yaml.load(f.read(), yaml.FullLoader)
    add_jieba_word(d)


def init_asr():
    f = open(os.path.join("asr", "asrConfig.yml"), 'r', encoding='utf-8')
    d = yaml.load(f.read(), yaml.FullLoader)
    return d
