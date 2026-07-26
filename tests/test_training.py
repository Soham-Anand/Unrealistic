import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import math
from src.training.scheduler import CosineScheduler


def test_warmup():
    sched = CosineScheduler(max_steps=1000, warmup_steps=100, peak_lr=3e-4)
    lr = sched(50)
    assert 0 < lr < 3e-4


def test_peak():
    sched = CosineScheduler(max_steps=1000, warmup_steps=100, peak_lr=3e-4)
    lr = sched(100)
    assert abs(lr - 3e-4) < 1e-6


def test_decay():
    sched = CosineScheduler(max_steps=1000, warmup_steps=100, peak_lr=3e-4)
    lr_200 = sched(200)
    lr_800 = sched(800)
    assert lr_200 > lr_800
