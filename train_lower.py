import argparse
import json
import os
import pickle
import random
import re
import shutil
import sys
import time
from copy import deepcopy
from glob import glob
from math import sqrt

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from einops import einsum
from scipy.stats import ttest_rel
from torch.distributions.categorical import Categorical
from torch.nn.init import xavier_uniform_
from torch.nn.utils.rnn import pad_sequence
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import Timer, bp_shelf, perm_by_height as perm_parts
from constants import BIN_HEIGHT, BIN_WIDTH


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=43, help="seed of the experiment")
    parser.add_argument("--cuda", type=int, default=-1)

    parser.add_argument('--dataset', choices=['G200', 'G100'], default='G100')

    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)

    parser.add_argument('--mha-layers', type=int, default=3, help='# of mha layers')
    parser.add_argument('--embed-dim', type=int, default=32)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument("--logit-scale", type=float, default=10)
    parser.add_argument("--load", type=str, default="", help='model checkpoint')
    parser.add_argument('--load-step', type=int, default=-1)
    parser.add_argument('--num-instances', type=int, default=16, help='# of instances to use')
    parser.add_argument('--num-test', type=int, default=100, help='# of test instances')
    parser.add_argument('--pomo', type=int, default=10, help='# of pomo start points')
    parser.add_argument('--pomo-entropy', type=float, default=1e-3, help='pomo entropy coefficient')
    parser.add_argument('--save-interval', type=int, default=100, help='checkpoint save interval')

    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
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


class MHA_Layer(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_hidden_dim=None, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        if mlp_hidden_dim is None:
            mlp_dim = (embed_dim, embed_dim * 4, embed_dim)
        else:
            mlp_dim = (embed_dim,) + mlp_hidden_dim + (embed_dim,)
        self.mlp = nn.Sequential(
            make_mlp(*mlp_dim, dropout=dropout, act=nn.Tanh),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.mha(x, x, x)[0]
        x = x + self.mlp(x)
        return x


def make_param(*dim):
    if len(dim) == 1:
        return nn.Parameter(torch.randn(dim[0], dtype=torch.float32) / sqrt(dim[0]))
    else:
        x = torch.empty(*dim, dtype=torch.float32)
        xavier_uniform_(x)
        return nn.Parameter(x)


def _mha_qx_batch(q, x):
    """
    q: H x D1 x D2
    x: B x N x (H x D1)
    -> B x H x N x D2
    """
    h, d1, d2 = q.shape
    b, n, _ = x.shape
    return torch.matmul(x.view(b, n, h, 1, d1), q).view(b, n, h, d2).permute(0, 2, 1, 3)


def _mha_qk_batch(q, k):
    """
    q: B x H x M x D
    k: B x H x N x D
    -> B x H x M x N
    """
    return torch.matmul(q, k.permute(0, 1, 3, 2)) / sqrt(k.size(3))


def _mha_ov_batch(o, v):
    """
    o: H x D1 x D2
    v: B x H x N x D1
    -> B x N x D2
    """
    return einsum(o, v, 'h d1 d2, b h n d1 -> b n d2')
    # return torch.matmul(v.unsqueeze(3), o.unsqueeze(1)).squeeze(3).sum(1)


class Model(nn.Module):
    def __init__(self, embed_dim=32, num_mha_layers=3, num_heads=4, logit_scale=10, dropout=0.1):
        super().__init__()
        self.bp = bp_shelf
        self.encoder = nn.Sequential(
            nn.Linear(2, embed_dim),
            *(MHA_Layer(embed_dim, num_heads, dropout=dropout) for _ in range(num_mha_layers)),
        )
        self.context_placeholder = make_param(embed_dim)
        head_dim = embed_dim // num_heads
        self.q_1 = make_param(num_heads, head_dim * 2, head_dim)
        self.k_1 = make_param(num_heads, head_dim, head_dim)
        self.v_1 = make_param(num_heads, head_dim, head_dim)
        self.o_1 = make_param(num_heads, head_dim, embed_dim)
        self.q_2 = make_param(1, embed_dim, embed_dim)
        self.k_2 = make_param(1, embed_dim, embed_dim)
        self.logit_scale = logit_scale

    def forward(self, x, sample, return_sol=False):
        x, parts = zip(*x)
        x = torch.stack(x)
        x = self.encoder(x)
        x_avg = x.mean(1)
        k_1 = _mha_qx_batch(self.k_1, x)
        v_1 = _mha_qx_batch(self.v_1, x)
        k_2 = _mha_qx_batch(self.k_2, x)
        last_index = None
        visited_mask = torch.zeros(*x.shape[:2], dtype=torch.bool, device=x.device)
        log_prob = 0
        sol = []
        iota = range(len(x_avg))
        for _ in range(len(parts[0])):
            context = torch.hstack([x_avg, self.context_placeholder.view(1, -1).expand(len(x_avg), -1) if last_index is None else x[iota, last_index]])
            u = _mha_qk_batch(_mha_qx_batch(self.q_1, context[:, None, :]), k_1)
            mask = visited_mask[:, None, None, :]
            u = u.masked_fill(mask, -1e999)
            a = torch.softmax(u, -1)
            h = _mha_ov_batch(self.o_1, torch.matmul(a, v_1))
            u = _mha_qk_batch(_mha_qx_batch(self.q_2, h), k_2)
            logits = self.logit_scale * torch.tanh(u).masked_fill(mask, -1e999).view(len(x_avg), -1)
            prob = Categorical(logits=logits)
            if sample:
                index = prob.sample()
            else:
                index = logits.detach().argmax(1)
            # assert not torch.any(visited_mask[iota, index])
            log_prob = log_prob + prob.log_prob(index)
            visited_mask = visited_mask.clone()
            visited_mask[iota, index] = True
            sol.append(index.cpu().tolist())
            last_index = index
        sol = list(zip(*sol))
        cost = torch.tensor([
            len(self.bp([a[i] for i in b])) for a, b in zip(parts, sol)
        ], dtype=torch.float32, device=log_prob.device)
        if return_sol:
            return cost, sol
        return cost, log_prob

    def pomo(self, x, K, sample, return_sol=False, return_bp=False):
        x, parts = zip(*x)
        x = torch.stack(x)
        x = self.encoder(x)
        x_avg = x.mean(1)
        k_1 = _mha_qx_batch(self.k_1, x)
        v_1 = _mha_qx_batch(self.v_1, x)
        k_2 = _mha_qx_batch(self.k_2, x)
        k_1_ = k_1.repeat_interleave(K, 0)
        v_1_ = v_1.repeat_interleave(K, 0)
        k_2_ = k_2.repeat_interleave(K, 0)
        B = len(x_avg)
        N = len(parts[0])
        iota = range(B)
        iota_k = [i for i in iota for _ in range(K)]
        x_avg_k = x_avg.repeat_interleave(K, 0).view(B * K, -1)

        context = torch.hstack([x_avg, self.context_placeholder.view(1, -1).expand(B, -1)])
        u = _mha_qk_batch(_mha_qx_batch(self.q_1, context[:, None, :]), k_1)
        a = torch.softmax(u, -1)
        h = _mha_ov_batch(self.o_1, torch.matmul(a, v_1))
        u = _mha_qk_batch(_mha_qx_batch(self.q_2, h), k_2)
        logits = torch.log_softmax(self.logit_scale * torch.tanh(u).view(B, -1), 1)
        init_entropy = Categorical(probs=logits).entropy()
        log_prob, indices = torch.topk(logits, K, 1)
        log_prob = log_prob.view(-1)
        indices = indices.view(-1)
        sol = torch.zeros(B * K, N, dtype=torch.long, device=x.device)
        sol[:, 0] = last_index = indices
        visited_mask = torch.zeros(B * K, N, dtype=torch.bool, device=x.device)
        visited_mask[range(B * K), last_index] = True

        for k in range(1, N):
            context = torch.hstack([x_avg_k, x[iota_k, last_index]])
            u = _mha_qk_batch(_mha_qx_batch(self.q_1, context[:, None, :]), k_1_)
            mask = visited_mask[:, None, None, :]
            u = u.masked_fill(mask, -1e999)
            a = torch.softmax(u, -1)
            h = _mha_ov_batch(self.o_1, torch.matmul(a, v_1_))
            u = _mha_qk_batch(_mha_qx_batch(self.q_2, h), k_2_)
            logits = self.logit_scale * torch.tanh(u).masked_fill(mask, -1e999).view(B * K, -1)
            prob = Categorical(logits=logits)
            if sample:
                index = prob.sample()
            else:
                index = logits.detach().argmax(1)
            # assert not torch.any(visited_mask[range(B * K), index])
            log_prob = log_prob + prob.log_prob(index)
            visited_mask = visited_mask.clone()
            visited_mask[range(B * K), index] = True
            sol[:, k] = last_index = index
        sol = sol.cpu().view(B, K, N).tolist()
        bp = [
            self.bp([a[i] for i in b]) for a, bs in zip(parts, sol) for b in bs
        ]
        cost = torch.tensor([len(i) for i in bp], dtype=torch.float32, device=log_prob.device)
        if return_sol:
            return cost, sol
        if return_bp:
            return cost, bp
        return cost, log_prob, init_entropy

    def pomo_batch(self, xs, K, sample, batch_size=32, return_sol=False, use_tqdm=False):
        device = next(self.parameters()).device
        ids, xs = zip(*sorted(enumerate(xs), key=lambda x: len(x[1])))
        ids = np.argsort(ids)
        bps = []
        sols = []
        for i in tqdm(range(0, len(xs), batch_size), disable=not use_tqdm):
            parts = xs[i:i + batch_size]
            x = pad_sequence([torch.tensor([[w / BIN_WIDTH, h / BIN_HEIGHT] for w, h in i], dtype=torch.float32, device=device) for i in parts], batch_first=True)
            if return_sol:
                bp, sol = self._pomo_batch(x, parts, K, sample, return_sol)
                bps += bp
                sols += sol
            else:
                bps += self._pomo_batch(x, parts, K, sample, return_sol)
        if return_sol:
            return [bps[i] for i in ids], [sols[i] for i in ids]
        return [bps[i] for i in ids]

    def _pomo_batch(self, x, parts, K, sample, return_sol):
        B, N = x.shape[:2]
        K = min(K, N)
        device = x.device
        lens = [len(i) for i in parts]
        init_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        for i, j in enumerate(lens):
            init_mask[i, j:] = True
        lens_ = torch.tensor(lens, dtype=torch.long, device=device)
        lens_k = lens_.repeat_interleave(K)
        x = self.encoder(x)
        x_avg = torch.stack([i[:j].mean(0) for i, j in zip(x, lens)])
        k_1 = _mha_qx_batch(self.k_1, x)
        v_1 = _mha_qx_batch(self.v_1, x)
        k_2 = _mha_qx_batch(self.k_2, x)
        k_1_ = k_1.repeat_interleave(K, 0)
        v_1_ = v_1.repeat_interleave(K, 0)
        k_2_ = k_2.repeat_interleave(K, 0)
        iota = range(B)
        iota_k = [i for i in iota for _ in range(K)]
        x_avg_k = x_avg.repeat_interleave(K, 0).view(B * K, -1)

        context = torch.hstack([x_avg, self.context_placeholder.view(1, -1).expand(B, -1)])
        u = _mha_qk_batch(_mha_qx_batch(self.q_1, context[:, None, :]), k_1)
        mask = init_mask[:, None, None, :]
        u = u.masked_fill(mask, -1e999)
        a = torch.softmax(u, -1)
        h = _mha_ov_batch(self.o_1, torch.matmul(a, v_1))
        u = _mha_qk_batch(_mha_qx_batch(self.q_2, h), k_2)
        logits = u.masked_fill(mask, -1e999).view(B, -1)
        # logits = torch.log_softmax(self.logit_scale * torch.tanh(u).view(B, -1), 1)
        # init_entropy = Categorical(probs=logits).entropy()
        log_prob, indices = torch.topk(logits, K, 1)
        # log_prob = log_prob.view(-1)
        indices = indices.view(-1)
        pomo_mask = indices >= lens_k
        sol = torch.zeros(B * K, N, dtype=torch.long, device=device)
        sol[:, 0] = last_index = indices
        visited_mask = init_mask.repeat_interleave(K, 0)
        visited_mask[range(B * K), last_index] = True

        for k in range(1, N):
            row_mask = (lens_k <= k) | pomo_mask
            context = torch.hstack([x_avg_k, x[iota_k, last_index]])
            u = _mha_qk_batch(_mha_qx_batch(self.q_1, context[:, None, :]), k_1_)
            mask = visited_mask[:, None, None, :]
            u = u.masked_fill(mask, -1e999)
            a = torch.softmax(u, -1)
            h = _mha_ov_batch(self.o_1, torch.matmul(a, v_1_))
            u = _mha_qk_batch(_mha_qx_batch(self.q_2, h), k_2_)
            logits = self.logit_scale * torch.tanh(u).masked_fill(mask, -1e999).view(B * K, -1)
            logits[row_mask] = 0
            prob = Categorical(logits=logits)
            if sample:
                index = prob.sample()
            else:
                index = logits.detach().argmax(1)
            # assert not torch.any(visited_mask[range(B * K), index][~row_mask])
            # log_prob = log_prob + prob.log_prob(index)
            visited_mask = visited_mask.clone()
            visited_mask[range(B * K), index] = True
            sol[:, k] = last_index = index
        ps = [[i, j[k, :len(i)].tolist()] for i, j, k in zip(parts, sol.cpu().view(B, K, N), (~pomo_mask).cpu().view(B, K))]
        if return_sol:
            ps = [[[self.bp([i[l] for l in k]), k] for k in j] for i, j in ps]
            return list(zip(*[min(i, key=lambda x:len(x[0])) for i in ps]))
        return [min([self.bp([i[l] for l in k]) for k in j], key=lambda x:len(x)) for i, j in ps]


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


def to_tensor(l, device):
    return [
        [torch.tensor([[w / BIN_WIDTH, h / BIN_HEIGHT] for w, h in i], dtype=torch.float32, device=device), i]
        for i in l
    ]


def main():
    args = parse_args()
    run_name = time.strftime('Lower_%y%m%d_%H%M%S')
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
    model = Model(embed_dim=args.embed_dim, num_mha_layers=args.mha_layers, logit_scale=args.logit_scale).to(device)
    if args.load:
        if '/' in args.load:
            path = args.load
        else:
            if args.load_step == -1:
                args.load_step = 1e8
            path = min(glob(f'log/{args.load}/pt/*.pt'), key=lambda x: abs(args.load_step - int(re.findall(r'(\d+)', x.rsplit('/', 1)[1])[0])))
        print(f'Load model from {path}')
        state_dict = torch.load(path, map_location=device)
        model.load_state_dict(state_dict)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, eps=1e-5)

    if args.dataset == 'G200':
        parts = [k for i in pickle.load(open('./dataset/G200_train_1000.pkl', 'rb')) for j in i for k in j]
        max_parts = 200
    elif args.dataset == 'G100':
        parts = [k for i in pickle.load(open('./dataset/G100_train_1000.pkl', 'rb')) for j in i for k in j]
        max_parts = 100
    else:
        raise ValueError

    test_set = to_tensor([random.sample(parts, max_parts) for _ in range(args.num_test)], device)
    baseline = deepcopy(model)
    timer = Timer(10)
    timer.step()
    with torch.no_grad():
        test_cost = baseline(test_set, sample=False)[0].cpu().numpy()
    test_challenge = -1
    test_p = 1

    tq = tqdm(range(args.total_timesteps), dynamic_ncols=True)
    try:
        for global_step in tq:
            data = to_tensor([random.sample(parts, max_parts) for _ in range(args.num_instances)], device)
            if args.pomo:
                cost, log_prob, init_entropy = model.pomo(data, K=args.pomo, sample=True)
                cost = cost.view(len(data), -1)
                loss_1 = ((cost - cost.mean(1).view(-1, 1)).view(-1) * log_prob).mean()
                loss_2 = -args.pomo_entropy * init_entropy.mean()
                loss = loss_1 + loss_2
                ratio = np.array([i.detach().cpu().item() for i in [loss_1, loss_2]])
                ratio /= ratio.sum()
                writer.add_scalars("losses/ratio", {
                    'REINFORCE': ratio[0],
                    'entropy': ratio[1],
                }, global_step)
                b_cost = cost.min(1).values
                cost = cost.mean(1)
            else:
                cost, log_prob = model(data, True)
                with torch.no_grad():
                    b_cost, _ = baseline(data, False)
                loss = ((cost - b_cost) * log_prob).mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            b_bins = b_cost.sum().cpu().item()
            bins = cost.detach().cpu().sum().item()
            loss = loss.detach().cpu().item()
            ref_bins = sum(len(bp_shelf(perm_parts(i[1]))) for i in data)
            if args.pomo:
                tq.set_description(f'{run_name} {test_cost.sum()}({test_challenge},{test_p:.4f}) {b_bins/ref_bins:.3f} {bins/ref_bins:.3f} {loss:.4f} {ratio[0]:.4f}')
            else:
                tq.set_description(f'{run_name} {test_cost.sum()}({test_challenge},{test_p:.4f}) {b_bins/ref_bins:.3f} {bins/ref_bins:.3f} {loss:.4f}')
            writer.add_scalar("charts/loss", loss, global_step)
            writer.add_scalar("charts/bin_ratio", bins / ref_bins, global_step)
            writer.add_scalar("charts/baseline", b_bins / ref_bins, global_step)
            timer.step()
            writer.add_scalar("charts/FPS", timer.fps(), global_step)

            if not args.debug and (global_step + 1) % args.save_interval == 0:
                torch.save(model.state_dict(), f'log/{run_name}/pt/{global_step+1}.pt')

            if global_step % 10 == 9:
                with torch.no_grad():
                    cost = model(test_set, sample=False)[0].cpu().numpy()
                p = ttest_rel(test_cost, cost, alternative='greater').pvalue
                if p < 0.05:
                    test_cost = cost
                    baseline = deepcopy(model)
                test_challenge = cost.sum()
                test_p = p
                writer.add_scalar("charts/test_bins", cost.sum(), global_step)
                writer.add_scalar("charts/test_p", p, global_step)

    except KeyboardInterrupt:
        if not args.debug:
            torch.save(model.state_dict(), f'log/{run_name}/pt/{global_step+1}.pt')
        raise
    finally:
        writer.close()


if __name__ == "__main__":
    main()
