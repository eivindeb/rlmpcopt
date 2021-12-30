import numpy as np
import os
import copy


def make_test_set(dataset_path, config_path, n_scenarios=10, seed=0):
    import os
    import pickle
    from gym_let_mpc.let_mpc import LetMPCEnv
    env = LetMPCEnv(config_path)
    env.seed(seed)
    if os.path.exists(dataset_path + ".pkl"):
        confirmation = input("A dataset already exists at the chosen path, you sure you want to overwrite? (y/n)")
        if confirmation != "y":
            print("Exiting")
            exit(0)
    dataset = env.create_dataset(n_scenarios)
    with open(dataset_path + ".pkl", "wb") as f:
        pickle.dump(dataset, f)
    print("Successfully created dataset at " + dataset_path)


class BaselineController:
    def __init__(self, obs_vars, eps=0, n_envs=1):
        self.obs_vars = []
        for var in obs_vars:
            for transform in var.get("transform", ["none"]):
                var_with_transform = copy.deepcopy(var)
                var_with_transform["transform"] = transform
                self.obs_vars.append(var_with_transform)

        self.eps = eps

        self.n_envs = n_envs

    def draw_action(self, state, **predictor_params):
        raise NotImplementedError

    def stop(self):
        pass

    def fit(self, dataset):
        pass

    def episode_start(self):
        pass

    def save(self, path, full_save=False):
        pass

    def reset(self, indices=None):
        pass


class EveryT(BaselineController):
    __name__ = "EveryT"

    def __init__(self, obs_vars, every_t, lqr=None, lqr_obs_idxs=None, eps=0, n_envs=1):
        self._every_t = every_t
        self._counter = np.array([1 for _ in range(n_envs)])
        self.lqr = lqr
        self.lqr_obs_idxs = lqr_obs_idxs
        self.__name__ = os.path.join(self.__name__, "Every{}".format(self._every_t))
        super().__init__(obs_vars, eps, n_envs=n_envs)

    def episode_start(self):
        self._counter = 1

    def draw_action(self, state, **predictor_params):
        if np.random.uniform() < self.eps:
            action = np.array([np.random.choice([0, 1]), np.random.normal(size=state.shape)])
        elif getattr(self, "lqr", None) is not None:
            lqr_obs = state[..., self.lqr_obs_idxs]
            if self.lqr.time_varying:
                k, x = lqr_obs[..., 0].astype(np.int32), np.atleast_2d(lqr_obs[..., 1:])
            else:
                k, x = None, lqr_obs
            action = np.concatenate([(self._counter % self._every_t == 0).astype(np.float32).reshape(-1, 1), self.lqr.get_action(x, k)], axis=1)
        else:
            action = (self._counter % self._every_t == 0).astype(np.float32)
        self._counter += 1
        return action

    def reset(self, indices=None):
        if indices is None:
            indices = list(range(self.n_envs))
        if isinstance(indices, int):
            indices = [indices]
        for i in indices:
            self._counter[i] = 1


class EveryTH(EveryT):
    __name__ = "EveryTH"

    def __init__(self, obs_vars, every_t, horizon, lqr=None, lqr_obs_idxs=None, eps=0, n_envs=1):
        self._horizon = horizon
        super().__init__(obs_vars, every_t=every_t, lqr=lqr, lqr_obs_idxs=lqr_obs_idxs, eps=eps, n_envs=n_envs)
        self.__name__ = os.path.join(self.__name__, "Every{}{}".format(self._every_t, self._horizon))
        self.last_execution_time = 0

    def draw_action(self, state, **predictor_params):
        action = super().draw_action(state, **predictor_params)
        return np.insert(action, 1, self._horizon, axis=-1)
    
    
def env_call_function(env, function_name, *args, **kwargs):
    from stable_baselines.common.vec_env import VecEnv
    from stable_baselines.bench import Monitor
    if issubclass(type(env), VecEnv):
        return env.env_method(function_name, *args, **kwargs)
    else:
        kwargs.pop("indices", None)
        if isinstance(env, Monitor):
            return getattr(env.env, function_name)(*args, **kwargs)
        else:
            return getattr(env, function_name)(*args, **kwargs)


def env_get_attr(env, attr_name, return_list=False):
    from stable_baselines.common.vec_env import VecEnv
    from stable_baselines.bench import Monitor
    if issubclass(type(env), VecEnv):
        attr = env.get_attr(attr_name)
        if not return_list:
            return attr[0]
        else:
            return attr
    else:
        if isinstance(env, Monitor):
            return getattr(env.env, attr_name)
        else:
            return getattr(env, attr_name)


def make_env(env_class, init_kw, rank=0, seed=0, info_kw=(), monitor=True, training=True, HER=False, her_norm=False, value_function=None):
    """
    Utility function for multiprocessed env.
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        from stable_baselines.common import set_global_seeds
        from stable_baselines.her import HERGoalEnvWrapper
        from stable_baselines.bench import Monitor

    def _init():
        env = env_class(**init_kw)
        env.training = training
        env.seed(seed + rank)
        if monitor:
            env = Monitor(env, filename=None, allow_early_resets=True, info_keywords=info_kw)
        if HER:
            env = HERGoalEnvWrapper(env, norm=her_norm)
        return env
    set_global_seeds(seed)
    return _init


if __name__ == "__main__":
    make_test_set(os.path.join("data", "new-set-25"), os.path.join("configs", "cart_pendulum.json"), 25, seed=0)