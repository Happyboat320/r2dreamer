import numpy as np

from .minecraft_base import MinecraftBase


BASIC_ACTIONS = {
    "noop": dict(),
    "attack": dict(attack=1),
    "turn_up": dict(camera=(-15, 0)),
    "turn_down": dict(camera=(15, 0)),
    "turn_left": dict(camera=(0, -15)),
    "turn_right": dict(camera=(0, 15)),
    "forward": dict(forward=1),
    "back": dict(back=1),
    "left": dict(left=1),
    "right": dict(right=1),
    "jump": dict(jump=1, forward=1),
    "place_dirt": dict(place="dirt"),
}


class MinecraftDiamond(MinecraftBase):
    def __init__(self, *, size, repeat, break_speed, seed):
        actions = {
            **BASIC_ACTIONS,
            "craft_planks": dict(craft="planks"),
            "craft_stick": dict(craft="stick"),
            "craft_crafting_table": dict(craft="crafting_table"),
            "place_crafting_table": dict(place="crafting_table"),
            "craft_wooden_pickaxe": dict(nearbyCraft="wooden_pickaxe"),
            "craft_stone_pickaxe": dict(nearbyCraft="stone_pickaxe"),
            "craft_iron_pickaxe": dict(nearbyCraft="iron_pickaxe"),
            "equip_stone_pickaxe": dict(equip="stone_pickaxe"),
            "equip_wooden_pickaxe": dict(equip="wooden_pickaxe"),
            "equip_iron_pickaxe": dict(equip="iron_pickaxe"),
            "craft_furnace": dict(nearbyCraft="furnace"),
            "place_furnace": dict(place="furnace"),
            "smelt_iron_ingot": dict(nearbySmelt="iron_ingot"),
        }
        self.items = [
            "log",
            "planks",
            "stick",
            "crafting_table",
            "wooden_pickaxe",
            "cobblestone",
            "stone_pickaxe",
            "iron_ore",
            "furnace",
            "iron_ingot",
            "iron_pickaxe",
            "diamond",
        ]
        self.rewards = [CollectReward(item) for item in self.items] + [HealthReward()]
        super().__init__(actions, self.items, size=size, repeat=repeat, break_speed=break_speed, seed=seed)

    def step(self, action):
        obs, _, done, info = super().step(action)
        if "error" in info:
            return obs, np.float32(0), done, info
        reward = sum(fn(obs, self.inventory) for fn in self.rewards)
        return obs, np.float32(reward), done, info

    def reset(self):
        obs = super().reset()
        for reward in self.rewards:
            reward(obs, self.inventory)
        return obs


class CollectReward:
    def __init__(self, item):
        self.item = item
        self.collected = False

    def __call__(self, obs, inventory):
        current = inventory[self.item]
        if obs["is_first"]:
            self.collected = current > 0
            return 0
        first_collection = current > 0 and not self.collected
        self.collected |= first_collection
        return int(first_collection)


class HealthReward:
    def __init__(self, scale=0.01):
        self.scale = scale
        self.previous = None

    def __call__(self, obs, inventory=None):
        health = float(obs["health"][0])
        if obs["is_first"]:
            self.previous = health
            return 0
        reward = self.scale * (health - self.previous)
        self.previous = health
        return reward
