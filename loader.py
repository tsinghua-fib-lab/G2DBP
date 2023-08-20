import json

import torch
from tqdm import tqdm

import pybp
from common import BIN_HEIGHT, BIN_WIDTH
from train_upper import Environment, EnvironmentGroup
from train_upper import Model as UpperModel
from train_upper import load_lower

assert pybp.__version__ >= '0.1.0', 'Please update your pybp version!'


def make_object(d: dict):
    class Obj:
        def __init__(self, d):
            self.__dict__.update(d)
    return Obj(d)


class Model:
    def __init__(self, upper_model_path, lower_model_path, cuda=-1):
        self.device = device = torch.device(f'cuda:{cuda}' if cuda >= 0 else 'cpu')
        self.args = args = make_object(json.load(open(f'{upper_model_path}/args.json')))
        self.model = model = UpperModel(args.embed_dim, no_share=args.no_share, no_layer_norm=args.no_layer_norm, hist_binary_size=args.hist_binary_size, logit_scale=args.logit_scale).to(device)
        model.load_state_dict(torch.load(f'{upper_model_path}/model.pt', map_location=device))
        model.eval()
        self.lower_model = load_lower(path=lower_model_path, device=device)

    def optimize(self, dataset, max_parts, show_tqdm=False, sample=False, num_steps=1000, reset_step=64, lower_pomo=10, lower_batch=100):
        env = EnvironmentGroup(
            [
                Environment(
                    orders=i,
                    max_parts=max_parts,
                    use_incumbent_reward=self.args.use_incumbent_reward,
                    hist_freq_size=self.args.hist_freq_size,
                    hist_binary_size=self.args.hist_binary_size,
                    reset_steps=reset_step,
                    device=self.device
                ) for i in dataset
            ],
            lower_method='RL',
            lower_model=self.lower_model,
            lower_pomo=lower_pomo,
            lower_batch=lower_batch,
            keep_detail=True,
        )
        usages = []
        next_obs = env.observe()
        with torch.no_grad():
            for _ in tqdm(range(num_steps), disable=not show_tqdm):
                usages.append([i.usage_avg_b for i in env.envs])
                action = self.model.get_action_and_value(next_obs, sample=sample, action_only=True)
                _, next_obs, _ = env.step(action)
        usages.append([i.usage_avg_b for i in env.envs])
        result = [
            [
                {
                    'order': i,
                    'item': sum((e.orders[k] for k in i), []),
                    'pos':j
                }
                for i, j in zip(
                    e.plan_b,
                    [
                        sorted([*i, *j] for i, j in zip(pybp.get_bin_out_fast_5(t, False, BIN_WIDTH, BIN_HEIGHT), t))
                        for t in e.detail_b
                    ]
                )
            ] for e in env.envs
        ]
        return usages, result
