import random
from copy import deepcopy
from math import log
from typing import List

from tqdm import tqdm

from common import (Solution, bp_shelf, perm_by_area_, perm_by_height_,
                    perm_by_width_)


class SA:
    def __init__(self, orders: List[List[int]], max_parts: int, perm_algo='height', device='cpu'):
        self.orders = orders
        self.max_parts = max_parts
        if perm_algo == 'height':
            self.perm_algo = perm_by_height_
        elif perm_algo == 'width':
            self.perm_algo = perm_by_width_
        elif perm_algo == 'area':
            self.perm_algo = perm_by_area_
        else:
            raise ValueError

    def bp_batch(self, batch):
        return [i and bp_shelf(self.perm_algo(i)) for i in batch]

    def search(self, sol: Solution, num_iter=1000, reset=100, topk=1000, use_tqdm=False, _track=None):
        assert len(sol.plan) > 1
        best_sol = deepcopy(sol)
        init_cost = best_cost = current_cost = sol.cost
        if _track is not None:
            _track.append(init_cost)
        with tqdm(range(num_iter), disable=not use_tqdm) as bar:
            for it in bar:
                if (it + 1) % reset == 0:
                    sol = deepcopy(best_sol)
                    current_cost = best_cost
                a, b = random.sample(range(len(sol.plan)), 2)
                a_plan, a_bins = sol.plan[a], sol.bins[a]
                b_plan, b_bins = sol.plan[b], sol.bins[b]
                la = sum(len(self.orders[i]) for i in a_plan)
                lb = sum(len(self.orders[i]) for i in b_plan)
                if random.random() < 0.5:
                    s = [i for i in a_plan if len(self.orders[i]) + lb <= self.max_parts]
                    if s:
                        if len(s) > topk:
                            s = random.sample(s, topk)
                        best = 1e999
                        best_i = -1
                        best_bins = None
                        p = [k for j in b_plan for k in self.orders[j]]
                        batch = sum(([
                            [k for j in a_plan if j != i for k in self.orders[j]],
                            p + self.orders[i]
                        ] for i in s), [])
                        for i, nb1, nb2 in zip(s, *[iter(self.bp_batch(batch))] * 2):
                            t = len(nb1) + len(nb2) - (len(a_bins) + len(b_bins))
                            if t < best:
                                best = t
                                best_i = i
                                best_bins = nb1, nb2
                        if best_bins and best < -log(random.random()) * (1 - (it + 1) / num_iter):
                            a_bins[:], b_bins[:] = best_bins
                            b_plan.append(best_i)
                            if not a_bins:
                                del sol.plan[a]
                                del sol.bins[a]
                            else:
                                a_plan.remove(best_i)
                            current_cost += best
                            if current_cost < best_cost:
                                best_sol = deepcopy(sol)
                                best_cost = current_cost
                                assert best_sol.cost == best_cost
                            bar.set_description(f'{init_cost} {best_cost} {current_cost}')
                else:
                    t = [[i, j] for i in a_plan for j in b_plan if la - len(self.orders[i]) + len(self.orders[j]) <= self.max_parts and lb +
                         len(self.orders[i]) - len(self.orders[j]) <= self.max_parts]
                    best = 1e999
                    best_xy = None
                    best_bins = None
                    if len(t) > topk:
                        t = random.sample(t, topk)
                    batch = sum(([
                        [k for j in a_plan if j != x for k in self.orders[j]] + self.orders[y],
                        [k for j in b_plan if j != y for k in self.orders[j]] + self.orders[x]
                    ] for x, y in t), [])
                    for (x, y), nb1, nb2 in zip(t, *[iter(self.bp_batch(batch))] * 2):
                        t = len(nb1) + len(nb2) - (len(a_bins) + len(b_bins))
                        if t < best:
                            best = t
                            best_xy = x, y
                            best_bins = nb1, nb2
                    if best_bins and best < -log(random.random()) * (1 - (it + 1) / num_iter):
                        a_bins[:], b_bins[:] = best_bins
                        x, y = best_xy
                        a_plan[a_plan.index(x)] = y
                        b_plan[b_plan.index(y)] = x
                        current_cost += best
                        if current_cost < best_cost:
                            best_sol = deepcopy(sol)
                            best_cost = current_cost
                            assert best_sol.cost == best_cost
                        bar.set_description(f'{init_cost} {best_cost} {current_cost}')
                if _track is not None:
                    _track.append(current_cost)
        return best_sol
