# Optimization of the Model Predictive Control Meta-Parameters Through Reinforcement Learning

This repository contains the models used and code to produce these models as presented in the [accompanying paper](https://arxiv.org/abs/2011.13365). 
The config folder contains configuration files for both environment and for the reinforcement learning algorithm. The data
folder contains the test-set as described and used in the paper, as well as a separate validation set used during training and
development of the reinforcement learning models. The models used to produce the results of the paper are included in the
models folder.

The code is based on the libraries [stable-baselines](https://stable-baselines.readthedocs.io/en/master), 
[do-mpc](https://www.do-mpc.com/en/latest/), and [gym-let-mpc](https://github.com/eivindeb/gym-letMPC). See their documentation
for explanations of their configurable parameters.

## Installation
Tested with Python 3.7 (<=3.7 is required for Tensorflow 1)
```shell
git clone https://github.com/eivindeb/rlmpcopt
cd rlmpcopt
pip install -r requirements.txt
```

## Training
Reinforcement learning models to tune the MPC can be trained as follows, e.g. for the "complete" policy with 4 parallel actors for lower wall-clock training time:
```shell
python train_model.py --rl_config_name rl_config --env_config_name cart_pendulum --model_name test --seed 0 --tb_port 6010 --n_env 4 --test_set_name validation-set-25
```

To train only one meta-parameter, e.g. the prediction horizon:
```shell
python train_model.py --rl_config_name rl_config_ah --env_config_name cart_pendulum_ah --model_name test --seed 0 --tb_port 6010 --n_env 4 --test_set_name validation-set-25
```

The reinforcement learning algorithm is configured according to the file provided as the "rl_config_name" argument, see 
[stable-baselines](https://stable-baselines.readthedocs.io/en/master) documentation for the meaning of these parameters.
## Evaluation
Reinforcement learning MPC tuning models and fixed MPC baselines can be evaluated on datasets through the evaluate.py script:

### RL Model
Provide the path to the model file, e.g. the following will evaluate one of the models from the paper, on the test set used in the paper:
```shell
python evaluate.py --model_path models/cart_pendulum/paper1/best/model.zip --seed 0 --test_set_name test-set-25
```

### Fixed MPC Baseline
To evaluate the fixed MPC baseline provide the string "MPC-T-H" as model_path, where T is the constant triggering interval
and H is the constant prediction horizon, e.g. for standard MPC computed every step with a horizon of 40:

```shell
python evaluate.py --model_path MPC-1-40 --seed 0 --test_set_name test-set-25 --env_config_path configs/cart_pendulum.json
```

## Citation
The paper can be cited as follows:
```shell
@article{bohn2021optimization,
  title={Optimization of the Model Predictive Control Meta-Parameters Through Reinforcement Learning},
  author={B{\o}hn, Eivind and Gros, Sebastien and Moe, Signe and Johansen, Tor Arne},
  journal={arXiv preprint arXiv:2111.04146},
  year={2021}
}
```


