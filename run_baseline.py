import argparse
import json
import os
import pickle
import random
import time
import sys

import numpy as np

from baseline_gga import GGA
from baseline_mg import min_group
from baseline_sa import SA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['real', 'G200', 'G100'], required=True)
    parser.add_argument('--instance', type=int, required=True)
    parser.add_argument('--upper', choices=['MG', 'GGA', 'SA'], required=True)
    parser.add_argument('--lower', choices=['height', 'width', 'area'], required=True)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()

    os.makedirs('results', exist_ok=True)

    name = f'{args.dataset}_{args.instance}_{args.upper}_{args.lower}'
    if os.path.exists(f'results/{name}.pkl'):
        print('Warning: file already exists')
        if not args.overwrite:
            exit()

    if args.dataset == 'real':
        ins = pickle.load(open('dataset/real.pkl', 'rb'))[args.instance]
        max_parts = 200
    elif args.dataset == 'G200':
        ins = pickle.load(open('dataset/G200_test_100.pkl', 'rb'))[args.instance]
        max_parts = 200
    elif args.dataset == 'G100':
        ins = pickle.load(open('dataset/G100_test_100.pkl', 'rb'))[args.instance]
        max_parts = 100
    else:
        raise ValueError
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = 'cpu'
    start_time = time.time()
    if args.upper == 'SA':
        sa = SA(
            orders=ins,
            max_parts=max_parts,
            perm_algo=args.lower,
            device=device
        )
        sol = min(GGA(
            orders=ins,
            max_parts=max_parts,
            pop_size=2,
            perm_algo=sa.perm_algo
        ).pop, key=lambda x: x.cost)
        _track = []
        sol = sa.search(sol, 1000, reset=100000, use_tqdm=True, _track=_track)
        pickle.dump([sol, _track], open(f'results/{name}.pkl', 'wb'))
    elif args.upper == 'MG':
        plan = min_group(ins, max_parts)
        sa = SA(
            orders=ins,
            max_parts=max_parts,
            perm_algo=args.lower,
            device=device
        )
        bps = sa.bp_batch([sum((ins[j] for j in i), []) for i in plan])
        pickle.dump([plan, bps], open(f'results/{name}.pkl', 'wb'))
    elif args.upper == 'GGA':
        gga = GGA(ins, max_parts, 16, perm_algo=SA([], 0, args.lower).perm_algo)
        gga.run_parallel(num_iters=1000, num_workers=args.workers, use_tqdm=True, save_name=f'results/{name}.pkl')
    else:
        raise ValueError
    t = time.time()
    with open(f'results/{name}.log', 'w') as f:
        end_time = time.time()
        json.dump({
            'start_time': start_time,
            'end_time': end_time,
            'duration': end_time - start_time,
            'command': ' '.join(sys.argv)
        }, f, indent=4)


if __name__ == '__main__':
    main()
