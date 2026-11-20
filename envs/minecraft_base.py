from contextlib import contextmanager

import gymnasium as gym
import numpy as np


@contextmanager
def _minerl_numpy_aliases():
    # MineRL recreates its legacy NumPy-dependent handlers on each reset.
    aliases = {"bool": bool, "int": int, "float": float}
    missing = {name: value for name, value in aliases.items() if name not in vars(np)}
    vars(np).update(missing)
    try:
        yield
    finally:
        for name in missing:
            delattr(np, name)


class MinecraftBase(gym.Env):
    def __init__(
        self,
        actions,
        log_items,
        *,
        seed,
        repeat=1,
        size=(64, 64),
        break_speed=100.0,
        sticky_attack=30,
        sticky_jump=10,
        pitch_limit=(-60, 60),
    ):
        self._repeat = repeat
        if break_speed != 1.0:
            sticky_attack = 0

        with _minerl_numpy_aliases():
            from .minecraft_minerl import NOOP_ACTION, MineRLEnv

            self._env = MineRLEnv(tuple(size), break_speed).make()
        self._world_random = np.random.default_rng(seed)
        self._inventory = {}

        # Observations
        self._inv_keys = [key for key in self._env.observation_space["inventory"].spaces if key != "log2"]
        self._log_items = tuple(log_items)
        self._max_inventory = None
        self._equip_enum = self._env.observation_space["equipped_items"]["mainhand"][
            "type"
        ].values.tolist()

        # Actions
        self._noop_action = NOOP_ACTION
        self._action_values = tuple({**NOOP_ACTION, **action} for action in actions.values())
        self.action_space = gym.spaces.Discrete(len(self._action_values))
        self._sticky_attack_length = sticky_attack
        self._sticky_attack_counter = 0
        self._sticky_jump_length = sticky_jump
        self._sticky_jump_counter = 0
        self._pitch_limit = pitch_limit
        self._pitch = 0

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                "image": gym.spaces.Box(0, 255, self._env.observation_space["pov"].shape, np.uint8),
                "inventory": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._inv_keys),), dtype=np.float32
                ),
                "inventory_max": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._inv_keys),), dtype=np.float32
                ),
                "equipped": gym.spaces.Box(
                    -np.inf, np.inf, (len(self._equip_enum),), dtype=np.float32
                ),
                "health": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "hunger": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "breath": gym.spaces.Box(-np.inf, np.inf, (1,), dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), dtype=np.bool_),
                "is_last": gym.spaces.Box(0, 1, (), dtype=np.bool_),
                "is_terminal": gym.spaces.Box(0, 1, (), dtype=np.bool_),
                **{
                    f"log_inventory/{item}": gym.spaces.Box(0, np.iinfo(np.int32).max, (), dtype=np.int32)
                    for item in self._log_items
                },
            }
        )

    def step(self, action):
        action = self._action(self._action_values[action].copy())
        following = self._noop_action.copy()
        for key in ("attack", "forward", "back", "left", "right"):
            following[key] = action[key]
        for act in [action] + [following] * (self._repeat - 1):
            obs, reward, done, info = self._env.step(act)
            if "error" in info:
                # MineRL supplies a sampled observation after a connection failure.
                last = dict(self._last_obs, is_first=False, is_last=True, is_terminal=False)
                return last, 0.0, True, info
            if done:
                break
        obs["is_first"] = False
        obs["is_last"] = bool(done)
        obs["is_terminal"] = bool(info.get("is_terminal", done))

        self._last_obs = self._obs(obs)
        return self._last_obs.copy(), reward, done, info

    @property
    def inventory(self):
        return self._inventory

    def reset(self):
        self._max_inventory = None

        self._env.seed(int(self._world_random.integers(2**31)), seed_spaces=False)
        with _minerl_numpy_aliases():
            obs = self._env.reset()
        obs["is_first"] = True
        obs["is_last"] = False
        obs["is_terminal"] = False
        self._last_obs = self._obs(obs)

        self._sticky_attack_counter = 0
        self._sticky_jump_counter = 0
        self._pitch = 0
        return self._last_obs.copy()

    def _obs(self, obs):
        self._inventory = dict(obs["inventory"])
        self._inventory["log"] += self._inventory.pop("log2")
        inventory = np.array([self._inventory[k] for k in self._inv_keys], np.float32)
        if self._max_inventory is None:
            self._max_inventory = inventory
        else:
            self._max_inventory = np.maximum(self._max_inventory, inventory)
        index = self._equip_enum.index(obs["equipped_items"]["mainhand"]["type"])
        equipped = np.zeros(len(self._equip_enum), np.float32)
        equipped[index] = 1.0
        life = obs["life_stats"]
        return {
            "image": obs["pov"],
            "inventory": inventory,
            "inventory_max": self._max_inventory.copy(),
            "equipped": equipped,
            "health": np.float32([life["life"]]) / 20,
            "hunger": np.float32([life["food"]]) / 20,
            "breath": np.float32([life["air"]]) / 300,
            "is_first": obs["is_first"],
            "is_last": obs["is_last"],
            "is_terminal": obs["is_terminal"],
            **{f"log_inventory/{item}": np.int32(self._inventory[item]) for item in self._log_items},
        }

    def close(self):
        self._env.close()

    def _action(self, action):
        if self._sticky_attack_length:
            if action["attack"]:
                self._sticky_attack_counter = self._sticky_attack_length
            if self._sticky_attack_counter > 0:
                action["attack"] = 1
                action["jump"] = 0
                self._sticky_attack_counter -= 1
        if self._sticky_jump_length:
            if action["jump"]:
                self._sticky_jump_counter = self._sticky_jump_length
            if self._sticky_jump_counter > 0:
                action["jump"] = 1
                action["forward"] = 1
                self._sticky_jump_counter -= 1
        if self._pitch_limit and action["camera"][0]:
            lo, hi = self._pitch_limit
            if not (lo <= self._pitch + action["camera"][0] <= hi):
                action["camera"] = (0, action["camera"][1])
            self._pitch += action["camera"][0]
        return action
