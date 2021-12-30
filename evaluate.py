import numpy as np
import tensorflow as tf

if type(tf.contrib) != type(tf): tf.contrib._warning = None
import tqdm
from stable_baselines.common.vec_env import VecNormalize, DummyVecEnv, VecEnv, SubprocVecEnv, VecEnvWrapper
from stable_baselines.lqr.policy import LQR
from stable_baselines.common.policies import RLMPCPolicy, AHETMPCLQRPolicy
import pickle
from gym_let_mpc.let_mpc import LetMPCEnv
from utils import env_get_attr, make_env, env_call_function, EveryTH
import copy
import os
import json


def evaluate_on_test_set(env_config_path, set_path, model, measures=(), seed=None, writer=None, timestep=None, render=False, render_kw=None, env_kw=None, deterministic=True):
    if writer is not None:
        assert timestep is not None

    if render_kw is None:
        render_kw = {}

    if not hasattr(model, "test_env"):  # TODO: set 0 action noise std, set LQR weights
        if env_kw is None:
            env_kw = {}
        env_kw.update({"environment": {"reward": {"normalize": {"std": 1.0, "mean": 0.0}}}})
        if model.n_envs > 1:
            env = SubprocVecEnv([make_env(LetMPCEnv, {"config_path": env_config_path, "config_kw": env_kw, "d": 1}, rank_i, info_kw=measures, seed=0 if seed is None else seed, monitor=True) for rank_i in range(model.n_envs)], reset_on_done=False)
        else:
            env = LetMPCEnv(env_config_path, config_kw=env_kw)#gym.make(env_name)
            env.seed(seed)
        if isinstance(model.env, VecNormalize):
            if model.n_envs == 1:
                env = DummyVecEnv([lambda: env], reset_on_done=False)
            env = VecNormalize(env, norm_reward=False)
        model.test_env = env

    if hasattr(model.act_model, "lqr") and getattr(model.act_model, "lqr") is None:
        obs_module_idxs = env_get_attr(model.test_env, "obs_module_idxs", False)
        lqr = env_call_function(model.test_env, "get_lqr", indices=0)
        if isinstance(lqr, list):
            lqr = lqr[0]
        cfg = env_get_attr(model.test_env, "config", False)
        model.act_model.time_varying = time_varying_lqr = cfg["lqr"].get("type", "time-invariant") == "time-varying"
        model.act_model.lqr = LQR(lqr.A, lqr.B, lqr.Q, lqr.R, time_varying_lqr)
        model.act_model.lqr_obs_idxs = [o_i == "lqr" for o_i in obs_module_idxs]
        original_system = None
        if getattr(model.act_model, "update_env_lqr", False):
            Q = model.act_model.lqr.get_numeric_value("Q")
            R = model.act_model.lqr.get_numeric_value("R")
            env_call_function(model.test_env, "update_lqr", Q=Q, R=R)
    elif getattr(model.act_model, "time_varying", False):
        time_varying_lqr = True
        original_system = {"A": model.act_model.lqr.get_numeric_value("A"), "B": model.act_model.lqr.get_numeric_value("B")}
    else:
        time_varying_lqr = False

    if hasattr(model.train_model, "lqr") and False:
        env_call_function(model.test_env, "update_lqr", **{"Q": model.train_model.lqr.get_numeric_value("Q"), "R": model.train_model.lqr.get_numeric_value("R")})

    if hasattr(model, "policy") and (issubclass(model.policy, RLMPCPolicy) or issubclass(model.policy, AHETMPCLQRPolicy)):
        original_last_horizon = np.copy(model.act_model.last_horizon)
        model.act_model.last_horizon = None

    norm = isinstance(model.test_env, VecNormalize)

    if norm:
        model.test_env.training = False
        model.test_env.obs_rms = model.env.obs_rms
        model.test_env.clip_mask = model.env.clip_mask

    if set_path.endswith(".npy"):
        scenarios = list(np.load(set_path, allow_pickle=True))
    elif set_path.endswith(".pkl"):
        with open(set_path, "rb") as f:
            scenarios = list(pickle.load(f))
    else:
        raise ValueError("Invalid test set file.")

    scenario_count = len(scenarios)
    if scenario_count < model.n_envs:
        scenarios = [scenarios[0] for i in range(model.n_envs)]
        scenario_count = model.n_envs
    test_pbar = tqdm.tqdm(total=len(scenarios), desc="Evaluating Agent")
    res = {measure: [0] * scenario_count for measure in measures}
    res["tot_rew"] = [0] * scenario_count

    active_envs = [True for i in range(model.n_envs)]
    env_scen_i = [i for i in range(model.n_envs)]
    test_done = False

    if (isinstance(model.test_env, VecEnv) or isinstance(model.test_env, VecEnvWrapper)) and model.n_envs > 1:
        scen_init = [scenarios.pop(0) for i in range(model.n_envs)]
        obs = model.test_env.reset(**{k: [scen[k] for scen in scen_init] for k in scen_init[0].keys()})
        if getattr(model.act_model, "time_varying", False):
            As, Bs = zip(*model.test_env.env_method("get_linearized_mpc_model_over_prediction"))
            if model.test_env.num_envs == 1:
                As, Bs = As[0], Bs[0]
            else:
                As, Bs = list(As), list(Bs)
            model.act_model.lqr.set_numeric_value({"A": As, "B": Bs})
    else:
        scen_init = [scenarios.pop(0)]
        obs = model.test_env.reset(**scen_init[0])
        if getattr(model.act_model, "time_varying", False):
            if isinstance(model.test_env, VecEnv) or isinstance(model.test_env, VecEnvWrapper):
                As, Bs = zip(*model.test_env.env_method("get_linearized_mpc_model_over_prediction"))
                if model.test_env.num_envs == 1:
                    As, Bs = As[0], Bs[0]
                else:
                    As, Bs = list(As), list(Bs)
            else:
                As, Bs = model.test_env.get_linearized_mpc_model_over_prediction()
            model.act_model.lqr.set_numeric_value({"A": As, "B": Bs})

    done = [False for _ in range(model.n_envs)]

    while not test_done:
        actions, _ = model.predict(obs, mask=done, deterministic=deterministic)

        obs, rew, done, info = model.test_env.step(actions)


        if not isinstance(model.test_env, VecEnv):
            done, info, rew = [done], [info], [rew]

        for i, env_done in enumerate(done):
            if time_varying_lqr and active_envs[i] and "As" in info[i]:
                As, Bs = info[i]["As"], info[i]["Bs"]
                model.act_model.lqr.set_numeric_value({"A": As, "B": Bs}, indices=i)
            if active_envs[i]:
                res["tot_rew"][env_scen_i[i]] += rew[i]
                for measure in measures:
                    if isinstance(info[i][measure], dict):
                        for state, outcome in info[i][measure].items():
                            if "{}/{}".format(measure, state) not in res:
                                res["{}/{}".format(measure, state)] = [[]]
                            res["{}/{}".format(measure, state)][env_scen_i[i]].append(outcome)
                    else:
                        res[measure][env_scen_i[i]] += info[i][measure]
            if env_done:
                if len(scenarios) > 0 or active_envs[i]:
                    if render is not None:
                        if render == "all" or (render == "success" and info[i]["success"]["all"]) or (render == "fail" and info[i].get("termination", None) != "steps"):
                            ep_render_kw = copy.deepcopy(render_kw)
                            if "save_path" in ep_render_kw:
                                if not os.path.exists(render_kw["save_path"]):
                                    os.makedirs(render_kw["save_path"])
                                ep_render_kw["save_path"] = os.path.join(ep_render_kw["save_path"], str(env_scen_i[i]) + ".png")
                            if isinstance(model.test_env, VecEnv) or isinstance(model.test_env, VecEnvWrapper):
                                model.test_env.render(indices=i, **ep_render_kw)
                            else:
                                model.test_env.render(**ep_render_kw)
                    if len(scenarios) > 0:
                        scenario = scenarios.pop(0)
                        env_scen_i[i] = (scenario_count - 1) - len(scenarios)
                        if hasattr(model, "reset"):
                            model.reset(indices=i)
                        if isinstance(model.test_env, VecEnv) or isinstance(model.test_env, VecEnvWrapper):
                            r_obs = model.test_env.reset(indices=i, **scenario)
                            if time_varying_lqr:
                                As, Bs = model.test_env.env_method("get_linearized_mpc_model_over_prediction", indices=i)[0]
                                model.act_model.lqr.set_numeric_value({"A": As,
                                                                       "B": Bs},
                                                                      indices=i)
                            obs[i] = r_obs
                        else:
                            obs = model.test_env.reset(**scenario)
                            if time_varying_lqr:
                                As, Bs = model.test_env.get_linearized_mpc_model_over_prediction()
                                model.act_model.lqr.set_numeric_value({"A": As, "B": Bs})
                    else:
                        active_envs[i] = False

                    test_pbar.update(1)

        if len(scenarios) == 0:
            test_done = not any(active_envs)

    if time_varying_lqr and original_system is not None:
        model.act_model.lqr.set_numeric_value(original_system)

    if hasattr(model, "policy") and (issubclass(model.policy, RLMPCPolicy) or issubclass(model.policy, AHETMPCLQRPolicy)):
        model.act_model.last_horizon = original_last_horizon

    if writer is not None:
        summaries = []
        for measure, measure_v in res.items():
            if isinstance(measure_v, dict):
                for state, v in measure_v.items():
                    summaries.append(
                        tf.Summary.Value(tag="test_set/{}_{}".format(measure, state), simple_value=np.nanmean(v)))
            else:
                summaries.append(
                    tf.Summary.Value(tag="test_set/{}".format(measure), simple_value=np.nanmean(measure_v)))
        writer.add_summary(tf.Summary(value=summaries), round(timestep, -2))
    else:
        for measure, measure_v in res.items():
            print(measure)
            if isinstance(measure_v, dict):
                for state, v in measure_v.items():
                    print("\t{}:\t{}".format(state, np.nanmean(np.array(v))))
            else:
                print("\t{}".format(np.nanmean(measure_v)))

    return res

if __name__ == "__main__":
    import argparse
    import stable_baselines
    import utils

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--seed", required=False, default=None)
    parser.add_argument("--test_set_name", required=True)
    parser.add_argument("--env_config_path", required=False)

    args = parser.parse_args()
    test_set_path = os.path.join("data", args.test_set_name + ".pkl")
    seed = args.seed
    if seed == "None":
        seed = None
    else:
        seed = int(seed)

    env_config_path = args.env_config_path

    if args.model_path.startswith("MPC"):
        assert args.env_config_path is not None

        class MockEnv:
            def __init__(self):
                pass

        class MockModel:
            def __init__(self, controller, n_envs):
                self.controller = controller
                self.env = MockEnv()
                self.n_envs = n_envs
                self.train_model = None
                self.act_model = controller

            def predict(self, state, **predict_kws):
                return self.controller.draw_action(state), None

            def reset(self, indices=None):
                self.controller.reset(indices)

        every_t = int(args.model_path.split("-")[1])
        horizon = int(args.model_path.split("-")[2])

        with open(env_config_path, "r") as f:
            env_cfg = json.load(f)

        loaded_model = MockModel(EveryTH(env_cfg["environment"]["observation"]["variables"], every_t, horizon, n_envs=1), n_envs=1)
        if every_t == 1:
            del loaded_model.controller.lqr
    else:
        if env_config_path is None:
            env_config_path = os.path.join(os.path.dirname(args.model_path), "..", "env_config.json")
        if os.path.exists(os.path.join(os.path.dirname(args.model_path), "obs_rms.pkl")):
            env = stable_baselines.common.vec_env.DummyVecEnv([utils.make_env(LetMPCEnv, {"config_path": env_config_path})])
            env = VecNormalize(env, norm_reward=False)
            env.load_running_average(os.path.dirname(args.model_path))
            obs_module_idxs = env_get_attr(env, "obs_module_idxs", False)
            env.clip_mask = [o_i != "lqr" for o_i in obs_module_idxs]
        else:
            print("WARNING: Not using normalization during evaluation")
            env = None
        loaded_model = stable_baselines.PPO2.load(args.model_path, env=env, seed=seed)
        loaded_model.n_envs = 1  # TODO: check if this is necessary

        model_lqr = getattr(loaded_model.act_model, "lqr", None)
        if model_lqr is not None:  # Load learned LQR weights
            model_lqr.set_weights(loaded_model.sess.run([p for p in loaded_model.params if p.name == 'model/LQR_weights:0'][0]))

    res = evaluate_on_test_set(env_config_path, test_set_path, loaded_model, measures=["reward/base", "reward/computation", "reward/constraint"], seed=seed, deterministic=True)