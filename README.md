# JaxCQL
A simple and modular implementation of the [Conservative Q Learning](https://arxiv.org/abs/2006.04779)
and [Soft Actor Critic](https://arxiv.org/abs/1812.05905) algorithm in Jax and Flax.

This repository is a reimplementation of [my other codebase of the same algorithms in Pytorch](https://github.com/young-geng/CQL).


## Installation

1. Install and use the included Ananconda environment
```
$ conda env create -f environment.yml
$ source activate JaxCQL
```
You'll need to [get your own MuJoCo key](https://www.roboti.us/license.html) if you want to use MuJoCo.

2. Add this repo directory to your `PYTHONPATH` environment variable.
```
export PYTHONPATH="$PYTHONPATH:$(pwd)"
```

## Run Experiments
You can run SAC experiments using the following command:
```
python -m JaxCQL.sac_main \
    --env 'HalfCheetah-v2' \
    --logging.output_dir './experiment_output'
```
All available command options can be seen in JaxCQL/conservative\_sac_main.py and JaxCQL/conservative_sac.py.


You can run CQL experiments using the following command:
```
python -m JaxCQL.conservative_sac_main \
    --env 'halfcheetah-medium-v0' \
    --logging.output_dir './experiment_output'
```

All available command options can be seen in JaxCQL/sac_main.py and JaxCQL/sac.py.


## Visualize Experiments
You can visualize the experiment metrics with viskit:
```
python -m viskit './experiment_output'
```
and simply navigate to [http://localhost:5000/](http://localhost:5000/)


## Weights and Biases Online Visualization Integration
This codebase can also log to [W&B online visualization platform](https://wandb.ai/site). To log to W&B, you first need to set your W&B API key environment variable:
```
export WANDB_API_KEY='YOUR W&B API KEY HERE'
```
Then you can run experiments with W&B logging turned on:
```
python -m JaxCQL.conservative_sac_main \
    --env 'halfcheetah-medium-v0' \
    --logging.output_dir './experiment_output' \
    --logging.online
```

## Results of Running JaxCQL on D4RL Environments
In order to save your time and compute resources, I've done a sweep of JaxCQL on certain
D4RL environments with various min Q weight values. [The results can be seen here](https://wandb.ai/ygx/JaxCQL--jax_cql_gym_sweep_1).
You can choose the environment to visualize by filtering on `env`. The results for each `cql.cql_min_q_weight` on each `env`
is repeated and average across 3 random seeds.


## Credits
The project organization is inspired by [TD3](https://github.com/sfujim/TD3).
The SAC implementation is based on [rlkit](https://github.com/vitchyr/rlkit).
THe CQL implementation is based on [CQL](https://github.com/aviralkumar2907/CQL).
The viskit visualization is taken from [viskit](https://github.com/vitchyr/viskit), which is taken from [rllab](https://github.com/rll/rllab).
