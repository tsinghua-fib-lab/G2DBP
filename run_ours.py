import argparse
import json
import os
import pickle
import sys
import time
from copy import deepcopy

import torch
from tqdm import tqdm

from train_upper import (Model, Environment, EnvironmentGroup, load_lower,
                         transpose)


def make_object(d: dict):
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)
    return Obj(d)


def slide_min(arr):
    ret = []
    m = 1e999
    for i in arr:
        m = min(m, i)
        ret.append(m)
    return ret


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['real', 'G200', 'G100'])
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--cuda', type=int, default=-1)
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--reset-step', type=int, default=64)
    parser.add_argument('--lower', choices=['height', 'width', 'area', 'RL'], default='height')
    parser.add_argument('--lower-pomo', type=int, default=10)
    parser.add_argument('--lower-batch', type=int, default=100)
    opts = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    device = torch.device(f'cuda:{opts.cuda}' if opts.cuda >= 0 else 'cpu')
    if opts.dataset == 'real':
        data = pickle.load(open('dataset/real.pkl', 'rb'))
        max_parts = 200
        model_path = './pretrained/Upper_200'
    elif opts.dataset == 'G200':
        data = pickle.load(open('dataset/G200_test_100.pkl', 'rb'))[:10]
        max_parts = 200
        model_path = './pretrained/Upper_200'
    elif opts.dataset == 'G100':
        data = pickle.load(open('dataset/G100_test_100.pkl', 'rb'))[:10]
        max_parts = 100
        model_path = './pretrained/Upper_100'
    else:
        raise ValueError

    args = make_object(json.load(open(f'{model_path}/args.json')))
    model = Model(args.embed_dim, no_share=args.no_share, no_layer_norm=args.no_layer_norm, hist_binary_size=args.hist_binary_size, logit_scale=args.logit_scale).to(device)
    model.load_state_dict(torch.load(f'{model_path}/model.pt', map_location=device))
    model.eval()

    name = f'results/{opts.dataset}_ours_{opts.lower}'
    print(name)

    if os.path.exists(name + '.done') != opts.overwrite:
        if opts.overwrite:
            print('Error: nothing to overwrite')
        else:
            print('Error: will overwrite')
        exit()

    if opts.lower == 'RL':
        lower_model = load_lower(path=f'pretrained/Lower_{max_parts}', device=device)
    else:
        lower_model = None
    log = {}
    start_time = time.time()
    env = EnvironmentGroup(
        [
            Environment(
                orders=i,
                max_parts=max_parts,
                use_incumbent_reward=args.use_incumbent_reward,
                hist_freq_size=args.hist_freq_size,
                hist_binary_size=args.hist_binary_size,
                reset_steps=opts.reset_step,
                device=device
            ) for i in data
        ],
        lower_method=opts.lower,
        lower_model=lower_model,
        lower_pomo=opts.lower_pomo,
        lower_batch=opts.lower_batch,
        lower_records=None,
        lower_detail=None,
    )
    usages = []
    log = [[deepcopy(i.orders) for i in env.envs], usages]
    next_obs = env.observe()
    with torch.no_grad():
        for _ in tqdm(range(opts.steps)):
            usages.append(transpose([[deepcopy(i.usage), deepcopy(i.plan)] for i in env.envs]))
            action = model.get_action_and_value(next_obs, sample=True, action_only=True)
            reward, next_obs, next_done = env.step(action)
    with open(f'{name}.log', 'w') as f:
        end_time = time.time()
        json.dump({
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'command': ' '.join(sys.argv)
        }, f, indent=4)
    pickle.dump(log, open(name + '.pkl', 'wb'))
    open(name + '.done', 'w').close()


if __name__ == '__main__':
    main()
