import random
from copy import deepcopy

import numpy as np
import torch

import tools


def save(path, agent, trainer):
    weights = {"agent_state_dict": agent.state_dict()}
    archive = path.parent / "checkpoints"
    archive.mkdir(exist_ok=True)
    _save(archive / f"step_{trainer.step}.pt", weights)

    # Only the rolling checkpoint carries replay and the rest of the training state.
    state = {
        **weights,
        "optims_state_dict": tools.recursively_collect_optim_state_dict(agent),
        "agent_training": agent.training_state_dict(),
        "trainer": trainer.state_dict(),
        "replay": trainer.replay_buffer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(agent.device) if agent.device.type == "cuda" else None,
        },
    }
    _save(path, state)
    print(f"Saved checkpoint at step {trainer.step}: {path}")


def _save(path, state):
    # Keep the previous checkpoint readable if the process stops during the write.
    temporary = path.with_suffix(".pt.tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path, agent, trainer):
    # Read replay from disk in batches while the model and optimizer are restored normally.
    state = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if "trainer" not in state:
        raise ValueError(f"Checkpoint has no training progress to resume: {path}")
    agent.load_state_dict(state["agent_state_dict"])
    optimizers = state["optims_state_dict"]
    if agent.device.type == "cpu":
        # Optimizer tensors can stay on the same device during load. Detach them
        # from the mapping so replacing latest.pt can release the old replay file.
        optimizers = deepcopy(optimizers)
    tools.recursively_load_optim_state_dict(agent, optimizers)
    agent.load_training_state_dict(state["agent_training"])
    trainer.load_state_dict(state["trainer"])
    trainer.replay_buffer.load_state_dict(state["replay"], trainer.train_envs.env_num)

    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if agent.device.type == "cuda" and rng["cuda"] is not None:
        torch.cuda.set_rng_state(rng["cuda"], agent.device)
    print(f"Resumed checkpoint at step {trainer.step}: {path}")
