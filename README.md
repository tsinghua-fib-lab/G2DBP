# Learning to Solve Grouped 2D Bin Packing Problems in Manufacturing Industry

This repo contains the source code and datasets for our KDD'23 ADS track paper under review.

## Requirements

```
python>=3.8
einops==0.6.0
numpy==1.21.2
ray==1.10.0
scipy==1.7.3
setproctitle==1.2.3
torch==1.13.0
tqdm==4.62.3
```

And install `pybp` by running
```bash
pip install git+https://github.com/John-Ao/pybp
```

## Datasets
The training set and testing set of the synthetic datasets `G200` and `G100` are in the `./dataset` folder. Due to privacy concerns, the `Real` dataset is not included. Instead, the synthetic dataset `G200` can be used, which is generated with statistics similar to `Real`.

## Baselines
The baseline methods can be run with `run_baseline.py`.

For example, to test `GGA-H` on the first instance of dataset `G100`, use
```bash
python run_baseline.py --upper GGA --lower height --dataset G100 --instance 0
```

## Our method
To train the lower agent on `G200` and `G100`, use
```bash
python train_lower.py --dataset G200 --embed-dim 32 --num-heads 4 --pomo 10 --cuda 0
python train_lower.py --dataset G100 --embed-dim 32 --num-heads 4 --pomo 10 --cuda 0
```

To train the upper agent on `G200` and `G100`, use
```bash
python train_upper.py --dataset G200 --lower RL --embed-dim 32 --cuda 0
python train_upper.py --dataset G100 --lower RL --embed-dim 32 --cuda 0
```

We provide checkpoints for pre-trained models on `G200` and `G100`. 
To test our method on the first 10 test instances of dataset `G100` and `G200`, use
```bash
python run_ours.py --dataset G100 --lower RL --steps 1000 --cuda 0
python run_ours.py --dataset G200 --lower RL --steps 1000 --cuda 0
```
