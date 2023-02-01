import random
import time
from collections import deque, namedtuple

import pybp

from constants import BIN_HEIGHT, BIN_WIDTH


def perm_by_height_(xy):
    xy.sort(key=lambda x: (-x[1], x[0]))
    return xy


def perm_by_width_(xy):
    xy.sort(key=lambda x: (-x[0], x[1]))
    return xy


def perm_by_area_(xy):
    xy.sort(key=lambda x: (-x[0] * x[1], -x[1], x[0]))
    return xy


def perm_by_height(xy):
    return sorted(xy, key=lambda x: (-x[1], x[0]))


def perm_by_width(xy):
    return sorted(xy, key=lambda x: (-x[0], x[1]))


def perm_by_area(xy):
    return sorted(xy, key=lambda x: (-x[0] * x[1], -x[1], x[0]))


def bp_shelf(xy):
    return pybp.get_bin_out_fast_3(xy, False, BIN_WIDTH, BIN_HEIGHT)[1]


def bp_max_rect(xy):
    return pybp.get_bin_out_fast_4(xy, False, BIN_WIDTH, BIN_HEIGHT)[1]


def transpose(arr):
    return [list(i) for i in zip(*arr)]


class Timer():
    def __init__(self, n=10):
        self.queue = deque([], n)

    def step(self):
        self.queue.append(time.time())

    def fps(self):
        assert len(self.queue) > 1
        return (len(self.queue) - 1) / (self.queue[-1] - self.queue[0])


class Solution(namedtuple('Solution', ['plan', 'bins'])):
    def shuffle(self):
        i = list(range(len(self.bins)))
        random.shuffle(i)
        self.bins[:] = [self.bins[i] for i in i]
        self.plan[:] = [self.plan[i] for i in i]

    @property
    def cost(self):
        return sum(len(i) for i in self.bins)
