import pickle
import random
from copy import deepcopy
from typing import List

import numpy as np
import ray
from tqdm import tqdm

from common import Solution, bp_shelf


def eps():
    return random.random() / 5 - .1


@ray.remote
class RayWorker:
    def __init__(self, gga):
        self.gga = gga

    def cross(self, i, j):
        self = self.gga
        all_ind = set(range(len(self.orders)))
        a, b = self.pop[i], self.pop[j]
        a.shuffle()
        b.shuffle()
        a_i, a_j = sorted(random.sample(range(len(a.plan) + 1), 2))
        b_i, b_j = sorted(random.sample(range(len(b.plan) + 1), 2))
        a_1, a_2, a_3 = a.plan[:a_i], a.plan[a_i:a_j], a.plan[a_j:]
        b_1, b_2, b_3 = b.plan[:b_i], b.plan[b_i:b_j], b.plan[b_j:]
        s_1 = {j for i in a_2 for j in i}
        s_2 = {j for i in b_2 for j in i}
        c_1 = [[j for j in i if j not in s_2] for i in a_1] + deepcopy(b_2) + [[j for j in i if j not in s_2] for i in a_3]
        c_1 = [i for i in c_1 if i]
        c_2 = [[j for j in i if j not in s_1] for i in b_1] + deepcopy(a_2) + [[j for j in i if j not in s_1] for i in b_3]
        c_2 = [i for i in c_2 if i]
        m_1 = list(all_ind - {j for i in c_1 for j in i})
        m_2 = list(all_ind - {j for i in c_2 for j in i})
        random.shuffle(m_1)
        random.shuffle(m_2)
        # c1_cost, c_1 = self.greedy_insert_(c_1, m_1)
        # c2_cost, c_2 = self.greedy_insert_(c_2, m_2)
        c_1 = self.topk_insert_(self.plan2sol(c_1), m_1, sample=True)
        c_2 = self.topk_insert_(self.plan2sol(c_2), m_2, sample=True)
        _track = [a.cost, b.cost, c_1.cost, c_2.cost]
        # _s = f'{a.cost} x {b.cost} -> {c_1.cost} {c_2.cost}'
        self.local_search(c_1)
        self.local_search(c_2)
        # print(f'{_s} -> {c_1.cost} {c_2.cost}')
        _track += [c_1.cost, c_2.cost]
        c = c_1 if c_1.cost < c_2.cost + eps() else c_2
        return i, (c if c.cost < a.cost + eps() else None), _track

    def set_pop(self, pop):
        self.gga.pop = pop


class GGA:
    def __init__(self, orders: List[List[int]], max_parts: int, pop_size: int, perm_algo=None):
        self.orders = orders
        self.max_parts = max_parts
        self.pop_size = pop_size
        self.perm_algo = perm_algo
        self.bp_call = 0
        self.orders_bp = [self.bp(i) for i in orders]
        assert pop_size > 1
        self.pop = self.init_pop()

    def bp(self, parts) -> List[float]:
        self.bp_call += 1
        return bp_shelf(self.perm_algo(parts))

    def bp_(self, parts) -> List[float]:
        if not parts:
            return []
        self.bp_call += 1
        return bp_shelf(self.perm_algo(parts))

    def init_pop(self) -> List[Solution]:
        ind = list(range(len(self.orders)))
        pop = []
        for _ in range(self.pop_size):
            random.shuffle(ind)
            pop.append(self.greedy_init(ind))
        return pop

    def plan2sol(self, plan):
        return Solution(plan, [self.bp([k for j in i for k in self.orders[j]]) for i in plan])

    def greedy_init(self, ind: List[int]) -> Solution:
        remain = ind[:]
        remain_len = [len(self.orders[i]) for i in remain]
        bins = []
        plan = []
        while True:
            best_i = -1
            best_u = 0
            best_us = None
            for i in range(1, len(remain) + 1):
                if sum(remain_len[:i]) > self.max_parts:
                    break
                us = self.bp([j for i in remain[:i] for j in self.orders[i]])
                u = np.mean(us)
                if u > best_u:
                    best_i = i
                    best_u = u
                    best_us = us
            assert best_us is not None
            bins.append(best_us)
            plan.append(remain[:best_i])
            if best_i == len(remain):
                break
            remain = remain[best_i:]
            remain_len = remain_len[best_i:]
        return Solution(plan, bins)

    def greedy_insert_(self, sol: Solution, ind: List[int]):
        ps = [[j, [k for j in i for k in self.orders[j]]] for i, j in zip(*sol)]
        for i in ind:
            o = self.orders[i]
            best_b = self.orders_bp[i][:]
            best = (len(best_b), -min(best_b))
            best_j = -1
            for j, (b, p) in enumerate(ps):
                if len(o) + len(p) <= self.max_parts:
                    _p = p + o
                    _b = self.bp(_p)
                    x = (len(_b) - len(b), -(min(_b) - min(b)))
                    if x < best:
                        best = x
                        best_j = j
                        best_b = _b
                        best_p = _p
            if best_j == -1:
                sol.plan.append([i])
                ps.append([best_b, o[:]])
            else:
                sol.plan[best_j].append(i)
                ps[best_j][0] = best_b
                ps[best_j][1] = best_p
        sol.bins = [i[0] for i in ps]
        return sol

    def topk_insert_(self, sol: Solution, ind: List[int], k=10, sample=False):
        ps = [[j, [k for j in i for k in self.orders[j]]] for i, j in zip(*sol)]
        for i in ind:
            o = self.orders[i]
            best_b = self.orders_bp[i][:]
            best = (len(best_b), -min(best_b))
            best_j = -1
            if len(ps) <= k:
                js = range(len(ps))
            else:
                js = random.sample(range(len(ps)), k) if sample else [i[1] for i in sorted([min(j[0]), i] for i, j in enumerate(ps))[:k]]
            for j in js:
                b, p = ps[j]
                if len(o) + len(p) <= self.max_parts:
                    _p = p + o
                    _b = self.bp(_p)
                    x = (len(_b) - len(b), -(min(_b) - min(b)))
                    if x < best:
                        best = x
                        best_j = j
                        best_b = _b
                        best_p = _p
            if best_j == -1:
                sol.plan.append([i])
                ps.append([best_b, o[:]])
            else:
                sol.plan[best_j].append(i)
                ps[best_j][0] = best_b
                ps[best_j][1] = best_p
        sol.bins[:] = [i[0] for i in ps]
        return sol

    def _check(self, x):
        y = sorted(sum(x, []))
        assert y == list(range(len(self.orders))), (x, y)

    def _get_bins(self, plan):
        return sum(len(self.bp([k for j in i for k in self.orders[j]])) for i in plan)

    def local_search(self, sol: Solution, num_iter=1000, topk=1000):
        assert len(sol.plan) > 1
        for _ in range(num_iter):
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
                    for i in s:
                        nb1 = self.bp_([k for j in a_plan if j != i for k in self.orders[j]])
                        nb2 = self.bp(p + self.orders[i])
                        t = len(nb1) + len(nb2) - (len(a_bins) + len(b_bins))
                        if t < best:
                            best = t
                            best_i = i
                            best_bins = nb1, nb2
                    if best < 0:
                        a_bins[:], b_bins[:] = best_bins
                        b_plan.append(best_i)
                        if not a_bins:
                            del sol.plan[a]
                            del sol.bins[a]
                        else:
                            a_plan.remove(best_i)
            else:
                t = [[i, j] for i in a_plan for j in b_plan if la - len(self.orders[i]) + len(self.orders[j]) <= self.max_parts and lb + len(self.orders[i]) - len(self.orders[j]) <= self.max_parts]
                best = 1e999
                best_xy = None
                best_bins = None
                for x, y in random.sample(t, topk) if len(t) > topk else t:
                    nb1 = self.bp([k for j in a_plan if j != x for k in self.orders[j]] + self.orders[y])
                    nb2 = self.bp([k for j in b_plan if j != y for k in self.orders[j]] + self.orders[x])
                    t = len(nb1) + len(nb2) - (len(a_bins) + len(b_bins))
                    if t < best:
                        best = t
                        best_xy = x, y
                        best_bins = nb1, nb2
                if best < 0:
                    a_bins[:], b_bins[:] = best_bins
                    x, y = best_xy
                    a_plan[a_plan.index(x)] = y
                    b_plan[b_plan.index(y)] = x

    def run(self, num_iters: int, use_tqdm=False):
        assert all(len(i.plan) > 1 for i in self.pop)
        all_ind = set(range(len(self.orders)))
        ind = list(range(self.pop_size))
        with tqdm(range(num_iters), disable=not use_tqdm) as bar:
            for _ in bar:
                random.shuffle(ind)
                _bp_call = self.bp_call
                for i, j in zip(ind, ind[1:] + ind[:1]):
                    a, b = self.pop[i], self.pop[j]
                    a.shuffle()
                    b.shuffle()
                    a_i, a_j = sorted(random.sample(range(len(a.plan) + 1), 2))
                    b_i, b_j = sorted(random.sample(range(len(b.plan) + 1), 2))
                    a_1, a_2, a_3 = a.plan[:a_i], a.plan[a_i:a_j], a.plan[a_j:]
                    b_1, b_2, b_3 = b.plan[:b_i], b.plan[b_i:b_j], b.plan[b_j:]
                    s_1 = {j for i in a_2 for j in i}
                    s_2 = {j for i in b_2 for j in i}
                    c_1 = [[j for j in i if j not in s_2] for i in a_1] + deepcopy(b_2) + [[j for j in i if j not in s_2] for i in a_3]
                    c_1 = [i for i in c_1 if i]
                    c_2 = [[j for j in i if j not in s_1] for i in b_1] + deepcopy(a_2) + [[j for j in i if j not in s_1] for i in b_3]
                    c_2 = [i for i in c_2 if i]
                    m_1 = list(all_ind - {j for i in c_1 for j in i})
                    m_2 = list(all_ind - {j for i in c_2 for j in i})
                    random.shuffle(m_1)
                    random.shuffle(m_2)
                    # c1_cost, c_1 = self.greedy_insert_(c_1, m_1)
                    # c2_cost, c_2 = self.greedy_insert_(c_2, m_2)
                    c_1 = self.topk_insert_(self.plan2sol(c_1), m_1, sample=True)
                    c_2 = self.topk_insert_(self.plan2sol(c_2), m_2, sample=True)
                    print(f'{a.cost} x {b.cost} -> {c_1.cost} {c_2.cost}', end='')
                    self.local_search(c_1)
                    self.local_search(c_2)
                    print(f' -> {c_1.cost} {c_2.cost}')
                    c = c_1 if c_1.cost < c_2.cost + eps() else c_2
                    if c.cost < a.cost + eps():
                        self.pop[i] = c
                print(self.bp_call - _bp_call)
                if use_tqdm:
                    c = [i.cost for i in self.pop]
                    bar.set_description(f'{min(c)} {np.mean(c):.2f}±{np.std(c):.2f}')

    def run_parallel(self, num_iters: int, num_workers=4, use_tqdm=False, save_name=None):
        assert all(len(i.plan) > 1 for i in self.pop)
        ind = list(range(self.pop_size))
        workers = [RayWorker.remote(self) for _ in range(min(num_workers, self.pop_size))]
        pool = ray.util.ActorPool(workers)
        _track = []
        _costs = []
        _plans = []
        with tqdm(range(num_iters), disable=not use_tqdm) as bar:
            for _ in bar:
                random.shuffle(ind)
                for i, c, _t in pool.map_unordered(lambda a, ij: a.cross.remote(*ij), zip(ind, ind[1:] + ind[:1])):
                    if c is not None:
                        self.pop[i] = c
                    _track += _t
                ray.wait([i.set_pop.remote(self.pop) for i in workers])
                c = [i.cost for i in self.pop]
                _costs.append(c)
                _plans.append([deepcopy(i.plan) for i in self.pop])
                bar.set_description(f'{min(c)} {np.mean(c):.2f}±{np.std(c):.2f}')
                if save_name:
                    with open(save_name, 'wb') as f:
                        pickle.dump([_track, _costs, self.orders, _plans], f)
