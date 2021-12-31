import numpy as np
import os
import logging
import tensorboard
from tensorboard.program import TensorBoardPortInUseError
from tensorboard.util import tb_logging
import time
import json
from collections import deque

import stable_baselines
from stable_baselines.common import set_global_seeds, callbacks
from stable_baselines.common import schedules

import tensorflow as tf
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv, SubprocVecEnv
import shutil

from utils import make_env, env_get_attr, env_call_function

from evaluate import evaluate_on_test_set

from gym_let_mpc.let_mpc import LetMPCEnv


def save_model(model, type, step=None):
    global model_folder, norm

    path = os.path.join(model_folder, type)

    if step is not None:
        path = os.path.join(path, str(step))
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "model")
    model.save(path)

    if norm:
        model.env.save_running_average(os.path.dirname(path))


class CustomCallback(callbacks.BaseCallback):
    """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
        """

    def __init__(self, verbose=0):
        super(CustomCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """

        global last_save, model, last_ep_info, info_kw, env, stats, model_args, env_config_name, \
            norm, env, checkpoint_interval, test_interval, model_folder
        ep_info_buf = getattr(self.model, "ep_info_buf", [])

        if len(ep_info_buf) > 0:
            if last_ep_info != ep_info_buf[-1]:
                last_ep_info = ep_info_buf[-1]

                ep_rews = [ep_info['r'] for ep_info in ep_info_buf]
                mean_ep_rew = np.nan if len(ep_rews) == 0 else np.mean(ep_rews)
                stats["last_mean_reward"] = mean_ep_rew
                stats["ep_info_buf"] = list(ep_info_buf)

                if mean_ep_rew > stats["best_mean_reward"] or np.isnan(stats["best_mean_reward"]):
                    save_model(self.model, "best")
                    stats["best_mean_reward"] = mean_ep_rew

                info = {}
                for ep_info in ep_info_buf:
                    for k in ep_info.keys():
                        if k in info_kw:
                            if k not in info:
                                info[k] = []

                            info[k].append(ep_info[k])

                if self.locals["writer"] is not None:
                    summaries = []
                    for measure in info_kw:
                        summaries.append(
                            tf.Summary.Value(tag="ep_info/{}".format(measure), simple_value=np.nanmean(info[measure])))
                    self.locals["writer"].add_summary(tf.Summary(value=summaries), self.model.num_timesteps)

                elif self.locals["n_updates"] % 8 == 0 and self.locals["n_updates"] != 0:
                    for info_k, info_v in info.items():
                        print("\n{}:\n\t".format(info_k) + "\n\t".join(
                            ["{:<10s}{:.2f}".format(k, np.nanmean(v)) for k, v in info_v.items()]))

        elif len(stats["ep_info_buf"]) > len(ep_info_buf):
            setattr(self.model, "ep_info_buf", deque(stats["ep_info_buf"]))
            last_ep_info = stats["ep_info_buf"][-1]

        if test_interval is not None and self.model.num_timesteps - stats["last_test"] >= test_interval:
            render_path = os.path.join(model_folder, "render", str(self.model.num_timesteps))
            if not os.path.exists(render_path):
                os.makedirs(render_path)
            save_model(self.model, "test", self.model.num_timesteps)
            results = evaluate_on_test_set(env_config_path, os.path.join("data", test_set_name + ".pkl"), model,
                                           writer=self.locals["writer"],
                                           measures=info_kw,
                                           seed=0,
                                           timestep=self.model.num_timesteps,
                                           render="all",
                                           render_kw={"save_path": render_path},
                                           deterministic=not "etonly" in env_config_name)

            score_comps = {k: np.mean(v) for k, v in results.items() if
                           k.startswith("reward/")}
            total_score = sum([v for v in score_comps.values()])

            if stats.get("best_test", None) is None or total_score < stats["best_test"]:
                stats["best_test"] = total_score
                save_model(model, "best")
                #test_info = {k: np.mean(v).astype(np.float64) for k, v in results.items() if not k.startswith("reward")}
                test_info = {}
                test_info["step"] = self.model.num_timesteps
            stats["last_test"] = self.model.num_timesteps

        if checkpoint_interval is not None and self.model.num_timesteps - stats["last_checkpoint"] >= checkpoint_interval:
            save_model(self.model, "checkpoint")
            stats["last_checkpoint"] = self.model.num_timesteps

            with open(os.path.join(model_folder, "stats.json"), "w") as stats_file:
                json.dump(stats, stats_file)

        return True


def initialize():
    global model_name, env_name, training, env, stats, model_args, \
        config_path, last_ep_info, tb_port, last_save, model_folder, info_kw, model, algorithm, \
        norm, seed, env_config_path, info_kw, n_env, norm_load_kw, checkpoint_interval, test_interval

    last_ep_info = []

    with open(config_path) as config_file:
        model_args = json.load(config_file)

    if seed is not None:
        model_args["seed"] = seed

    model_folder = os.path.join("models", env_config_name, model_name)

    os.makedirs(model_folder, exist_ok=True)
    with open(os.path.join(model_folder, "config.json"), "w") as config_file:
        json.dump(model_args, config_file)
    shutil.copy2(env_config_path, os.path.join(model_folder, "env_config.json"))
    env_config_path = os.path.join(model_folder, "env_config.json")

    training_steps = int(model_args.pop("training_steps", 1e6))

    norm = model_args.pop("normalize_obs", False)

    checkpoint_interval = model_args.pop("checkpoint_interval", 5000)
    test_interval = model_args.pop("test_interval", 5000)

    seed = model_args.pop("seed", 0)
    env_d = model_args.pop("env_d", 1)  # frame skip
    if n_env > 1:
        env = SubprocVecEnv([make_env(LetMPCEnv, {"config_path": env_config_path, "d": env_d}, rank_i, info_kw=info_kw, seed=seed + rank_i, monitor=True) for rank_i in range(n_env)])
    else:
        env = LetMPCEnv(env_config_path)
        env.seed(seed)

    if norm:
        obs_variables = env_get_attr(env, "config", return_list=False)["environment"]["observation"]["variables"]
        obs_module_idxs = env_get_attr(env, "obs_module_idxs", False)
        if not (isinstance(env, SubprocVecEnv) or isinstance(env, DummyVecEnv)):
            _env = DummyVecEnv([lambda: env])
        else:
            _env = env
        mean_mask = [(obs_var["type"] == "constraint" and obs_var["value_type"] == "distance") or obs_module_idxs[o_i] == "lqr" for o_i, obs_var in enumerate(obs_variables)]
        var_mask = [o_i == "lqr" for o_i in obs_module_idxs]
        clip_mask = [o_i != "lqr" for o_i in obs_module_idxs]
        env = VecNormalize(_env, norm_reward=False, clip_reward=None, gamma=model_args.get("gamma", 0.99),
                           mean_mask=mean_mask, var_mask=var_mask, clip_mask=clip_mask)
    if model_args["policy"] == "LQRPolicy":
        lqr = env_call_function(env, "get_lqr", indices=0)
        if isinstance(lqr, list):
            lqr = lqr[0]
        model_args["policy_kwargs"].update({"A": lqr.A, "B": lqr.B, "Q": lqr.Q, "R": lqr.R, "std": 0.1,
                                            "obs_module_indices": env_get_attr(env, "obs_module_idxs", False)})
        if isinstance(lqr.A, list):
            model_args["policy_kwargs"]["time_varying"] = True
        if n_env > 1:
            model_args["policy_kwargs"]["n_lqr"] = n_env
        del model_args["policy_kwargs"]["layers"]["pi"]
    elif model_args["policy"] == "AHETMPCLQRPolicy":
        lqr = env.env_method("get_lqr", indices=0)[0]
        model_args["policy_kwargs"].update({"A": lqr.A, "B": lqr.B, "Q": lqr.Q, "R": lqr.R, "std": 0.1, "update_env_lqr": False,
                                            "obs_module_indices": env_get_attr(env, "obs_module_idxs", False)})
        if isinstance(lqr.A, list):
            model_args["policy_kwargs"]["time_varying"] = True
        model_args["policy_kwargs"]["n_lqr"] = n_env

    if os.path.exists(os.path.join(model_folder, "stats.json")):
        with open(os.path.join(model_folder, "stats.json")) as stats_file:
            stats = json.load(stats_file)
    else:
        stats = {"best_mean_reward": np.nan, "last_mean_reward": np.nan, "last_test": 0, "num_episodes": 0, "last_checkpoint": 0,
                 "ep_info_buf": []}

    for k, v in model_args.items():
        if isinstance(v, str) and v.startswith("Schedule"):
            args = v.split("_")[1:]
            type = "{}{}".format(args.pop(0), "Schedule")
            args = list(map(float, args))
            args.insert(0, training_steps)
            model_args[k] = getattr(schedules, type)(*args)

    model_args["model_class"] = getattr(stable_baselines, algorithm)

    tensorboard_folder = os.path.join(model_folder, "tensorboard")
    # Remove request prints etc.
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    tf_log = logging.getLogger("tensorflow")
    tf_log.setLevel(logging.ERROR)
    #np.seterr(all='raise')
    tb_started = False
    while not tb_started and tb_port is not None:
        try:
            # Start tensorboard
            tb = tensorboard.program.TensorBoard()
            # Need host argument when ipv6 is disabled
            tb.configure(argv=[None, "--host", "localhost", "--port", str(tb_port), "--logdir", tensorboard_folder])
            url = tb.launch()
            tb_started = True
            print("Launched tensorboard at {}".format(url))
        except TensorBoardPortInUseError:
            tb_port += 1

    tb_logging.get_logger().setLevel(logging.ERROR)

    model_args.update({"verbose": 1, "tensorboard_log": tensorboard_folder})

    alg = model_args.pop("model_class")
    model_args.pop("goal_selection_strategy", None)
    policy = getattr(stable_baselines.common.policies, model_args.pop("policy"))
    model = alg(policy, env, **model_args)

    set_global_seeds(seed)
    last_save = time.time()

    print("Training model {} for {} timesteps for environment config {} with args:".format(model_name, training_steps, env_config_name))
    for k, v in model_args.items():
        if k == "policy_kwargs":
            print(k)
            for kp, vp in v.items():
                if kp not in ["A", "B"]:
                    print("\t{}: {}".format(kp, vp))
        else:
            print("{}: {}".format(k, v))
    model.ep_info_buf = deque(maxlen=10)
    model.learn(total_timesteps=training_steps -  model.num_timesteps, log_interval=10, callback=CustomCallback())

    save_model(model, type="final")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--rl_config_name", required=True)
    parser.add_argument("--env_config_name", required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--seed", required=False, default=None)
    parser.add_argument("--tb_port", required=False)
    parser.add_argument("--n_env", required=True)
    parser.add_argument("--test_set_name", required=True)

    args = parser.parse_args()

    training = True
    n_env = int(args.n_env)
    env_config_name = args.env_config_name
    m_name = args.model_name
    seed = args.seed
    config_path = os.path.join("configs", args.rl_config_name + ".json")
    tb_port = args.tb_port
    if seed == "None":
        seed = None
    else:
        seed = int(seed)

    env_config_path = os.path.join("configs", env_config_name + ".json")
    test_set_name = args.test_set_name

    checkpoint_interval = 5000
    test_interval = 15000

    if seed is None:
        model_name = m_name
    else:
        model_name = "{}_s{}".format(m_name, seed)
    env = None
    stats = None
    algorithm = "PPO2"

    # Declare globals
    model_args = {}
    last_ep_info = None
    last_save = None
    model_folder = None
    info_kw = ["reward/base", "reward/computation", "reward/constraint"]
    model = None

    initialize()
