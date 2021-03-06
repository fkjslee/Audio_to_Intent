# reference: https://github.com/monologg/JointBERT
import argparse

from trainer import Trainer
from utils import init_logger, set_seed, get_args


def main(args):
    init_logger()
    set_seed(args)
    Trainer()


if __name__ == '__main__':
    args = get_args()
    main(args)
