import yaml
import torch
import random
from tqdm import tqdm, trange
import time

epochs = 5
for _ in trange(epochs, desc="out", position=0):
    x = [1, 2, 3]
    tq = tqdm(x, desc="in", position=0)
    for i, j in enumerate(tq):
        time.sleep(0.5)
