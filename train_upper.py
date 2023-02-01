import argparse
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from collections import deque
from copy import deepcopy
from distutils.util import strtobool
from glob import glob
from math import log

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.nn.utils.rnn import pack_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import (Timer, bp_shelf, perm_by_area, perm_by_height,
                    perm_by_width, transpose)
from constants import BIN_HEIGHT, BIN_WIDTH
from train_lower import Model as LowerModel

ACTION_MOVE = 0
ACTION_SWAP = 1


class Environment():
    def __init__(self, orders, max_parts, use_incumbent_reward, hist_freq_size, hist_binary_size, reset_steps, device, plan=None):
        self.max_parts = max_parts
        self.reset_steps = reset_steps
        self.step_cnt = 0
        self.device = device
        n = max(hist_binary_size, hist_freq_size) * 2
        self.history = deque([-1] * n, n)
        self.hist_binary_size = hist_binary_size
        self.hist_freq_size = hist_freq_size
        self.use_incumbent_reward = use_incumbent_reward
        orders = orders[:]
        if plan is None:
            orders.sort(key=lambda x: -len(x))
        self.orders = orders
        self.order_parts = [len(i) for i in orders]
        self.orders_norm = [[[x / BIN_WIDTH, y / BIN_HEIGHT] for x, y in perm_by_height(i)[::-1]] for i in orders]
        self.area_norm = [sum(x * y for x, y in i) for i in self.orders_norm]
        self.orders_norm = [torch.tensor(i, dtype=torch.float32, device=device) for i in self.orders_norm]
        if plan is None:
            l, map_ = zip(*sorted([(len(j), i) for i, j in enumerate(orders)], key=lambda x: (-x[0], x[1])))
            assert all(i <= max_parts for i in l)
            n = sum(l)
            for n in range((n + max_parts - 1) // max_parts, len(l)):
                bins = [[] for _ in range(n)]
                lb = [0] * n
                for id_, i in enumerate(l):
                    j = np.argmin(lb)
                    if lb[j] + i > max_parts:
                        break
                    lb[j] += i
                    bins[j].append(id_)
                else:
                    break
            else:
                assert False
            assert all(0 < i <= max_parts for i in lb)
            self.plan = [[map_[j] for j in i] for i in bins]
        else:
            self.plan = deepcopy(plan)

    def _init_get(self):
        g1 = self.orders
        g2 = [sum((self.orders[j] for j in i), []) for i in self.plan]
        return g1, g2

    def _init_set(self, u1, u2):
        self.orders_usage = torch.tensor(
            [[np.mean(i), max(i), min(i)] for i in u1],
            dtype=torch.float32, device=self.device
        )
        self.usage = u2
        self.usage_best = self.usage_avg = np.mean(sum(self.usage, []))
        self.plan_b = deepcopy(self.plan)
        self.usage_b = deepcopy(self.usage)
        self.usage_avg_b = self.usage_avg

    def observe(self):
        hist = []
        l = list(reversed(self.history))
        for i in range(len(self.plan)):
            h = [l[:self.hist_freq_size * 2].count(i) / self.hist_freq_size]
            for j in range(self.hist_binary_size):
                h.append(1 if l[j * 2] == i or l[j * 2 + 1] == i else 0)
            hist.append(h)
        return (
            self.orders_norm,
            self.orders_usage,
            self.order_parts,
            self.max_parts,
            deepcopy(self.plan),
            torch.tensor(hist, dtype=torch.float32, device=self.device),
            torch.tensor([[np.mean(i), max(i), min(i)] for i in self.usage], dtype=torch.float32, device=self.device),
        )

    def step(self, action):
        act_type, x1, y1, x2, y2 = action
        self.history.append(x1)
        self.history.append(x2)
        self.step_cnt += 1
        if act_type == ACTION_MOVE:
            self.plan[x1].remove(y1)
            self.plan[x2].append(y1)
        else:
            self.plan[x1].remove(y1)
            self.plan[x1].append(y2)
            self.plan[x2].remove(y2)
            self.plan[x2].append(y1)
        self._step_save = x1, x2

    def _step_get(self):
        return [sum((self.orders[j] for j in self.plan[i]), []) for i in self._step_save]

    def _step_set(self, u):
        x1, x2 = self._step_save
        self.usage[x1], self.usage[x2] = u
        old_usage = self.usage_avg
        self.usage_avg = np.mean(sum(self.usage, []))
        if self.usage_avg > self.usage_avg_b:
            self.plan_b = deepcopy(self.plan)
            self.usage_b = deepcopy(self.usage)
            self.usage_avg_b = self.usage_avg
        if self.use_incumbent_reward:
            reward = max(0, self.usage_avg - self.usage_best)
            self.usage_best = max(self.usage_avg, old_usage)
        else:
            reward = self.usage_avg - old_usage
        done = False
        if self.reset_steps and self.step_cnt % self.reset_steps == 0:
            self.plan = deepcopy(self.plan_b)
            self.usage = deepcopy(self.usage_b)
            self.usage_avg = self.usage_avg_b
            self.history.extend([-1] * len(self.history))
            done = True
        return reward * 100, self.observe(), done


def get_feasible_batch_pairs(order_parts, plan, max_parts):
    plan_parts = [sum(order_parts[j] for j in i) for i in plan]
    pairs = []
    for x1, i in enumerate(plan):
        for x2, m in enumerate(plan_parts):
            if x2 == x1:
                continue
            for y1 in i:
                n = order_parts[y1]
                if n + m <= max_parts:
                    pairs.append((x1, x2))
    pairs = set(pairs)
    for x1, i in enumerate(plan):
        n1 = plan_parts[x1]
        for x2 in range(x1 + 1, len(plan)):
            if (x1, x2) in pairs:
                continue
            n2 = plan_parts[x2]
            for y1 in i:
                m1 = order_parts[y1]
                found = False
                for y2 in plan[x2]:
                    m2 = order_parts[y2]
                    if n2 - m2 + m1 <= max_parts and n1 - m1 + m2 <= max_parts:
                        pairs.add((x1, x2))
                        found = True
                        break
                if found:
                    break
    return transpose(list(pairs))


def get_feasible_actions(order_parts, plan, max_parts, x1, x2):
    acts_1 = []
    n1 = sum(order_parts[j] for j in plan[x1])
    n2 = sum(order_parts[j] for j in plan[x2])
    for y1 in plan[x1]:
        n = order_parts[y1]
        if n + n2 <= max_parts and x1 != x2:
            acts_1.append([x1, y1, x2])
    acts_2 = []
    for y1 in plan[x1]:
        m1 = order_parts[y1]
        for y2 in plan[x2]:
            m2 = order_parts[y2]
            if n2 - m2 + m1 <= max_parts and n1 - m1 + m2 <= max_parts:
                acts_2.append([x1, y1, x2, y2])
    acts_all = (
        [(ACTION_MOVE, x1, y1, x2, -1) for x1, y1, x2 in acts_1]
        +
        [(ACTION_SWAP, x1, y1, x2, y2) for x1, y1, x2, y2 in acts_2]
    )
    assert acts_all
    return transpose(acts_1), transpose(acts_2), acts_all


class EnvironmentGroup():
    def __init__(self, envs, lower_method, lower_model, lower_pomo, lower_batch, lower_records, lower_detail=None):
        self.envs = envs
        env: Environment = envs[0]
        self.device = env.device
        if lower_method in ['height', 'RL']:
            self.perm_parts = perm_by_height
        elif lower_method == 'width':
            self.perm_parts = perm_by_width
        elif lower_method == 'area':
            self.perm_parts = perm_by_area
        else:
            raise NotImplementedError
        self.lower_method = lower_method
        self.lower_model = lower_model
        self.lower_detail = lower_detail
        self.lower_records = lower_records
        self.lower_pomo = lower_pomo
        self.lower_batch = lower_batch
        bps = self._usage_multiple([j for i in envs for j in i._init_get()])
        for i, e in enumerate(envs):
            e._init_set(*bps[i * 2:i * 2 + 2])

    def observe(self):
        return [i.observe() for i in self.envs]

    def step(self, actions):
        for i, j in zip(self.envs, actions):
            i.step(j)
        reward, obs, done = transpose([
            i._step_set(j)
            for i, j in zip(self.envs, self._usage_multiple([i._step_get() for i in self.envs]))
        ])
        return (
            torch.tensor(reward, dtype=torch.float32, device=self.device),
            obs,
            torch.tensor(done, dtype=torch.float32, device=self.device),
        )

    def _usage(self, batch):
        bps = [bp_shelf(self.perm_parts(i)) for i in batch]
        if self.lower_method == 'RL':
            to_handle = [[i, j, k] for i, (j, k) in enumerate(zip(bps, batch)) if sum(j) < len(j) - 1]
            if self.lower_detail is not None:
                self.lower_detail.extend([parts, bp, None] for parts, bp in zip(batch, bps) if sum(bp) >= len(bp) - 1)
            bps2 = to_handle and self.lower_model.pomo_batch([i[2] for i in to_handle], K=self.lower_pomo, sample=False, batch_size=self.lower_batch)
            for (i, bp, parts), bp2 in zip(to_handle, bps2):
                if self.lower_records is not None:
                    self.lower_records.append([len(bp), len(bp2)])
                if self.lower_detail is not None:
                    self.lower_detail.append([parts, bp, bp2])
                if len(bp2) < len(bp):
                    bps[i] = bp2
        return bps

    def _usage_multiple(self, arr):
        # assert all(type(i) is list for i in arr)
        l = [0] + np.cumsum([len(i) for i in arr]).tolist()
        bps = self._usage(sum(arr, []))
        return [bps[i:j] for i, j in zip(l, l[1:])]

    def get_mean_usage(self):
        return np.mean([np.mean(sum(i.usage, [])) for i in self.envs])

    def get_min_usage(self):
        return np.mean([min(sum(i.usage, [])) for i in self.envs])

    def get_total_bins(self):
        return sum(sum(len(j) for j in i.usage) for i in self.envs)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=43, help="seed of the experiment")
    parser.add_argument("--cuda", type=int, default=-1)

    parser.add_argument('--dataset', choices=['G200', 'G100'])
    parser.add_argument('--lower', choices=['height', 'width', 'area', 'RL'], default='RL', help='choice of lower permutation method')
    parser.add_argument("--lower-pomo", type=int, default=10)
    parser.add_argument("--lower-batch", type=int, default=100)

    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99, help="the discount factor gamma")
    parser.add_argument("--gae", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True, help="Use GAE for advantage computation")
    parser.add_argument("--gae-lambda", type=float, default=0.9, help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4, help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=8, help="the K epochs to update the policy")
    parser.add_argument("--ent-coef", type=float, default=0, help="coefficient of the entropy")
    parser.add_argument("--vf-coef", type=float, default=0.5, help="coefficient of the value function")
    parser.add_argument("--clip-coef", type=float, default=0.25, help="the surrogate clipping coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="the maximum norm for the gradient clipping")
    parser.add_argument('--norm-adv', action='store_true')

    parser.add_argument("--embed-dim", type=int, default=32, help='embedding size')
    parser.add_argument("--logit-scale", type=float, default=10)
    parser.add_argument("--hist-freq-size", type=int, default=10, help='history size')
    parser.add_argument("--hist-binary-size", type=int, default=3, help='history size')
    parser.add_argument('--use-incumbent-reward', action='store_true')
    parser.add_argument('--no-share', action='store_true')
    parser.add_argument('--no-layer-norm', action='store_true', default=True)

    parser.add_argument("--load", type=str, default="", help='model checkpoint')
    parser.add_argument('--load-step', type=int, default=-1)
    parser.add_argument('--num-instances', type=int, default=16, help='# of instances to use')
    parser.add_argument('--num-steps', type=int, default=4, help='env rollout steps')
    parser.add_argument('--reset-steps', type=int, default=32, help='steps before reset')
    parser.add_argument('--reset-env-steps', type=int, default=2000, help='steps before reset')
    parser.add_argument('--improve-env-steps', type=int, default=10, help='steps of initial improvement')
    parser.add_argument('--save-interval', type=int, default=50, help='checkpoint save interval')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    args.batch_size = args.num_instances * args.num_steps
    args.minibatch_size = args.batch_size // args.num_minibatches
    if args.minibatch_size == 0:
        print(f'Warning: divide minibatch: {args.batch_size} -> {args.num_minibatches}')
        args.minibatch_size = 1
        args.num_minibatches = args.batch_size
    return args


def make_mlp(*shape, dropout=0.1, act=nn.Tanh, sigma=0):
    ls = [nn.Linear(i, j) for i, j in zip(shape, shape[1:])]
    if sigma > 0:
        for l in ls:
            nn.init.orthogonal_(l.weight, 2**0.5)
            nn.init.constant_(l.bias, 0)
        nn.init.orthogonal_(ls[-1].weight, sigma)
    return nn.Sequential(
        *sum(([
            l,
            act(),
            nn.Dropout(dropout),
        ] for l in ls[:-1]), []),
        ls[-1]
    )


class Model(nn.Module):
    def __init__(self, embed_dim, no_share, no_layer_norm, hist_binary_size, logit_scale, dropout=0.1):
        super().__init__()
        self.hist_binary_size = hist_binary_size
        dim_order = 3
        dim_batch = embed_dim * 4 + 3 + 1 + hist_binary_size
        self.ln_order = nn.LayerNorm(dim_order)
        self.enc_order = make_mlp(dim_order, embed_dim * 2, embed_dim, dropout=dropout)
        self.gru_batch = nn.GRU(embed_dim, embed_dim, 1)
        self.ln_batch = nn.Sequential() if no_layer_norm else nn.LayerNorm(dim_batch)
        if no_share:
            self.ln_order_1 = nn.LayerNorm(dim_order)
            self.enc_order_1 = make_mlp(dim_order, embed_dim * 2, embed_dim, dropout=dropout)
            self.gru_batch_1 = nn.GRU(embed_dim, embed_dim, 1)
            self.ln_batch_1 = nn.Sequential() if no_layer_norm else nn.LayerNorm(dim_batch)
        self.no_share = no_share
        self.enc_batch = make_mlp(dim_batch, embed_dim * 2, embed_dim, dropout=dropout)
        self.batch_pair = nn.Sequential(
            make_mlp(dim_batch * 2, embed_dim * 2, embed_dim, 1, dropout=dropout, sigma=0.01),
            nn.Tanh()
        )
        self.s_o_in_b = nn.Sequential(
            make_mlp(dim_order + dim_batch, embed_dim, embed_dim, 1),
            nn.Tanh()
        )
        self.s_o_to_b = nn.Sequential(
            make_mlp(dim_order + dim_batch, embed_dim, embed_dim, 1),
            nn.Tanh()
        )
        self.s_o_to_o = nn.Sequential(
            make_mlp(dim_order * 2, embed_dim, embed_dim, 1),
            nn.Tanh()
        )
        self.s_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.s_beta = nn.Parameter(torch.tensor(0, dtype=torch.float32))
        self.critic = make_mlp(embed_dim, embed_dim, 1, dropout=dropout, sigma=1)
        self.logit_scale = logit_scale

    def _encode(self, obs):
        order, order_usage, _, _, plan, hist, plan_usage = obs
        o_2 = self.ln_order(order_usage)
        o_3 = self.enc_order(o_2)
        b_1 = [o_3[sorted(i)] for i in plan]
        _, h = self.gru_batch(pack_sequence(b_1, enforce_sorted=False))
        b_2 = self.ln_batch(torch.hstack([
            h[0],
            torch.stack([i.mean(0) for i in b_1]),
            torch.stack([i.max(0).values for i in b_1]),
            torch.stack([i.min(0).values for i in b_1]),
            hist,
            plan_usage
        ]))
        return o_2, b_2

    def _encode_1(self, obs):
        order, order_usage, _, _, plan, hist, plan_usage = obs
        o_2 = self.ln_order(order_usage)
        o_3 = self.enc_order_1(o_2)
        b_1 = [o_3[sorted(i)] for i in plan]
        _, h = self.gru_batch_1(pack_sequence(b_1, enforce_sorted=False))
        b_2 = self.ln_batch_1(torch.hstack([
            h[0],
            torch.stack([i.mean(0) for i in b_1]),
            torch.stack([i.max(0).values for i in b_1]),
            torch.stack([i.min(0).values for i in b_1]),
            hist,
            plan_usage
        ]))
        return o_2, b_2

    def get_value(self, obs):
        if self.no_share:
            return self.critic(torch.stack([self.enc_batch(self._encode_1(i)[-1]).mean(0) for i in obs]))
        return self.critic(torch.stack([self.enc_batch(self._encode(i)[-1]).mean(0) for i in obs]))

    def get_action_and_value(self, obs, action=None, sample=True, action_only=False):
        acts = []
        log_probs = []
        entropy = 0
        enc_bs = []
        for i, o in enumerate(obs):
            _, _, order_parts, max_parts, plan, _, _ = o
            enc_o, enc_b = self._encode(o)
            enc_bs.append(enc_b)
            pairs = get_feasible_batch_pairs(order_parts, plan, max_parts)
            logits_1 = self.batch_pair(torch.hstack([enc_b[pairs[0]], enc_b[pairs[1]]])) * self.logit_scale
            probs_1 = Categorical(logits=logits_1.view(-1))
            if action is None:
                index_1 = probs_1.sample() if sample else torch.argmax(logits_1.view(-1))
                x1, x2 = pairs[0][index_1], pairs[1][index_1]
            else:
                x1, x2 = action[i][1], action[i][3]
                index_1 = torch.tensor(list(zip(*pairs)).index((x1, x2)), dtype=torch.long, device=logits_1.device)
            f_acts_1, f_acts_2, f_acts_all = get_feasible_actions(order_parts, plan, max_parts, x1, x2)

            logits_2 = []
            if len(f_acts_1):
                logits_2.append((
                    -self.s_o_in_b(torch.hstack([enc_o[f_acts_1[1]], enc_b[f_acts_1[0]]]))
                    + self.s_o_to_b(torch.hstack([enc_o[f_acts_1[1]], enc_b[f_acts_1[2]]]))
                ).view(-1))
            if len(f_acts_2):
                logits_2.append((
                    self.s_alpha * (-self.s_o_in_b(torch.hstack([enc_o[f_acts_2[1]], enc_b[f_acts_2[0]]]))
                                    + self.s_o_to_b(torch.hstack([enc_o[f_acts_2[1]], enc_b[f_acts_2[2]]]))
                                    - self.s_o_in_b(torch.hstack([enc_o[f_acts_2[3]], enc_b[f_acts_2[2]]]))
                                    + self.s_o_to_b(torch.hstack([enc_o[f_acts_2[3]], enc_b[f_acts_2[0]]])))
                    + self.s_beta * self.s_o_to_o(torch.hstack([enc_o[f_acts_2[1]], enc_o[f_acts_2[3]]]))
                ).view(-1))
            logits_2 = torch.cat(logits_2) * self.logit_scale
            probs_2 = Categorical(logits=logits_2)
            if action is None:
                index_2 = probs_2.sample() if sample else torch.argmax(logits_2)
                acts.append(f_acts_all[index_2])
            else:
                act = action[i]
                index_2 = torch.tensor(f_acts_all.index(act), dtype=torch.long, device=logits_2.device)
                acts.append(act)
            if action_only:
                continue
            log_probs.append(probs_1.log_prob(index_1) + probs_2.log_prob(index_2))

            entropy = entropy + probs_1.entropy() / (log(len(logits_1)) or 1) + probs_2.entropy() / (log(len(logits_2)) or 1)
        if action_only:
            return acts
        return (
            action or acts,
            (torch.stack(log_probs) if sample else None),
            entropy / len(obs) / 2,
            self.get_value(obs) if self.no_share else self.critic(torch.stack([self.enc_batch(i).mean(0) for i in enc_bs])),
        )


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Dummy():
    def add_text(*_):
        pass

    def add_scalar(*_):
        pass

    def add_scalars(*_):
        pass

    def close(*_):
        pass


def make_object(d: dict):
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)
    return Obj(d)


def load_lower(path, device):
    args = make_object(json.load(open(f'{path}/args.json')))
    model = LowerModel(embed_dim=args.embed_dim, num_mha_layers=args.mha_layers, num_heads=args.num_heads).to(device)
    model.load_state_dict(torch.load(f'{path}/model.pt', map_location=device))
    return model.eval()


def main():
    args = parse_args()
    if args.dataset == 'G200':
        max_parts = 200
        data = pickle.load(open('dataset/G200_train_1000.pkl', 'rb'))
    elif args.dataset == 'G100':
        max_parts = 100
        data = pickle.load(open('dataset/G100_train_1000.pkl', 'rb'))
    else:
        raise ValueError
    run_name = time.strftime('PPO_%y%m%d_%H%M%S')
    print(f'Run: {run_name}')
    print(f'tensorboard --port 8888 --logdir log/{run_name}')
    if not args.debug:
        os.makedirs(f'log/{run_name}/code')
        os.makedirs(f'log/{run_name}/pt')
        for i in [__file__]:
            shutil.copy(i, f'log/{run_name}/code/')
        with open(f'log/{run_name}/code/run', 'w') as f:
            f.write('#!/bin/sh\npython ' + ' '.join(sys.argv))
        json.dump(vars(args), open(f'log/{run_name}/args.json', 'w'), indent=4)

    writer = Dummy() if args.debug else SummaryWriter(f"log/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    set_seed(args.seed)

    device = torch.device(f'cuda:{args.cuda}' if args.cuda >= 0 else 'cpu')
    # assert args.no_parts and args.no_layer_norm
    agent = Model(args.embed_dim, no_share=args.no_share, no_layer_norm=args.no_layer_norm, hist_binary_size=args.hist_binary_size, logit_scale=args.logit_scale).to(device)
    if args.load:
        if '/' in args.load:
            path = args.load
        else:
            if args.load_step == -1:
                args.load_step = 1e8
            path = min(glob(f'log/{args.load}/pt/*.pt'), key=lambda x: abs(args.load_step - int(re.findall(r'(\d+)', x.rsplit('/', 1)[1])[0])))
        print(f'Load model from {path}')
        state_dict = torch.load(path, map_location=device)
        agent.load_state_dict(state_dict)

    if args.lower == 'RL':
        lower_model = load_lower(path=f'pretrained/Lower_{args.dataset[1:]}', device=device)
    else:
        lower_model = None

    optimizer = optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)

    timer = Timer(10)
    timer.step()

    tq = tqdm(range(args.total_timesteps), dynamic_ncols=True)
    try:
        for global_step in tq:
            if global_step % args.reset_env_steps == 0:
                indices = random.sample(range(len(data)), args.num_instances)
                env = EnvironmentGroup(
                    [
                        Environment(
                            orders=data[i],
                            max_parts=max_parts,
                            use_incumbent_reward=args.use_incumbent_reward,
                            hist_freq_size=args.hist_freq_size,
                            hist_binary_size=args.hist_binary_size,
                            reset_steps=args.reset_steps,
                            device=device
                        ) for i in indices
                    ],
                    lower_method=args.lower,
                    lower_model=lower_model,
                    lower_pomo=args.lower_pomo,
                    lower_batch=args.lower_batch,
                    lower_records=None,
                )
                init_bins = env.get_total_bins()
                obs = [None] * args.num_steps
                actions = [None] * args.num_steps
                log_probs = torch.zeros((args.num_steps, args.num_instances), device=device)
                rewards = torch.zeros((args.num_steps, args.num_instances), device=device)
                dones = torch.zeros((args.num_steps, args.num_instances), device=device)
                values = torch.zeros((args.num_steps, args.num_instances), device=device)
                next_obs = env.observe()
                next_done = torch.zeros(args.num_instances, device=device)
                improve_steps = min(args.reset_env_steps, global_step // args.reset_env_steps * args.improve_env_steps)
                if improve_steps:
                    with torch.no_grad():
                        for _ in range(improve_steps):
                            action = agent.get_action_and_value(next_obs, action_only=True)
                            next_obs = env.step(action)[1]
                improved_bins = env.get_total_bins()
            for step in range(args.num_steps):
                obs[step] = next_obs
                dones[step] = next_done
                with torch.no_grad():
                    action, log_prob, _, value = agent.get_action_and_value(next_obs)
                    values[step] = value.view(-1)
                actions[step] = action
                log_probs[step] = log_prob

                reward, next_obs, next_done = env.step(action)
                rewards[step] = reward

            usage = env.get_mean_usage()
            usage_min = env.get_min_usage()
            bins = env.get_total_bins()
            r = reward.mean().cpu().item()
            tq.set_description(f'{run_name} {r:.4f} {improved_bins}({improve_steps},{improved_bins-init_bins:+d}) {bins}({bins-improved_bins:+d}) {usage:.4f} {usage_min:.4f}')
            writer.add_scalar("charts/usage", usage, global_step)
            writer.add_scalar("charts/usage_min", usage_min, global_step)
            writer.add_scalars("charts/bins", {
                'init': init_bins,
                'improve': improved_bins,
                'agent': bins
            }, global_step)
            writer.add_scalar("charts/reward", r, global_step)

            # bootstrap value if not done
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                if args.gae:
                    advantages = torch.zeros_like(rewards).to(device)
                    last_gae_lam = 0
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_nonterminal = 1.0 - next_done
                            next_values = next_value
                        else:
                            next_nonterminal = 1.0 - dones[t + 1]
                            next_values = values[t + 1]
                        delta = rewards[t] + args.gamma * next_values * next_nonterminal - values[t]
                        advantages[t] = last_gae_lam = delta + args.gamma * args.gae_lambda * next_nonterminal * last_gae_lam
                    returns = advantages + values
                else:
                    returns = torch.zeros_like(rewards).to(device)
                    for t in reversed(range(args.num_steps)):
                        if t == args.num_steps - 1:
                            next_nonterminal = 1.0 - next_done
                            next_return = next_value
                        else:
                            next_nonterminal = 1.0 - dones[t + 1]
                            next_return = returns[t + 1]
                        returns[t] = rewards[t] + args.gamma * next_nonterminal * next_return
                    advantages = returns - values

            # flatten the batch
            b_obs = sum(obs, [])
            b_log_probs = log_probs.reshape(-1)
            b_actions = sum(actions, [])
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = values.reshape(-1)

            # Optimizing the policy and value network
            clipfracs = []
            for epoch in range(args.update_epochs):
                b_inds = np.arange(args.batch_size)
                np.random.shuffle(b_inds)
                b_inds = b_inds.reshape(-1, args.minibatch_size)
                for mb_inds in b_inds:
                    _, new_log_prob, entropy, new_value = agent.get_action_and_value(
                        [b_obs[i] for i in mb_inds],
                        [b_actions[i] for i in mb_inds]
                    )
                    log_ratio = new_log_prob - b_log_probs[mb_inds]
                    if np.any(log_ratio.detach().cpu().numpy() >= 80):
                        print('Warning: log_ratio too big', log_ratio)
                        log_ratio = torch.clamp(log_ratio, None, 80)
                    ratio = log_ratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-log_ratio).mean()
                        approx_kl = ((ratio - 1) - log_ratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    v_loss = 0.5 * ((new_value.view(-1) - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - entropy_loss * args.ent_coef + v_loss * args.vf_coef

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                    optimizer.step()

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
            writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
            writer.add_scalar("losses/explained_variance", explained_var, global_step)
            timer.step()
            writer.add_scalar("charts/FPS", timer.fps(), global_step)
            ratio = np.array([i.item() for i in [pg_loss, entropy_loss * args.ent_coef, v_loss * args.vf_coef]])
            ratio = ratio / ratio.sum()
            writer.add_scalars("losses/ratio", {
                'policy': ratio[0],
                'entropy': ratio[1],
                'value': ratio[2],
            }, global_step)

            if not args.debug and (global_step + 1) % args.save_interval == 0:
                torch.save(agent.state_dict(), f'log/{run_name}/pt/{global_step+1}.pt')
    except KeyboardInterrupt:
        if not args.debug:
            torch.save(agent.state_dict(), f'log/{run_name}/pt/{global_step+1}.pt')
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
