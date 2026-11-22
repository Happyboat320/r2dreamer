## R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation

This repository provides the official implementation of [R2-Dreamer][r2dreamer] (ICLR 2026), a decoder-free, augmentation-free world model with redundancy-reducing regularization on latent representations. It also includes an efficient PyTorch DreamerV3 reproduction that trains **~5x faster** than a widely used [codebase][dreamerv3-torch], along with other baselines. Selecting R2-Dreamer via the config provides an additional **~1.6x speedup** over this baseline.

## Instructions

Install dependencies. This repository is tested with Ubuntu 24.04 and Python 3.11.

Installing via a virtual env like [uv](https://docs.astral.sh/uv/) is recommended. If you prefer Docker, follow [`docs/docker.md`](docs/docker.md).

```bash
# Select the environments you need, or use "all" to install everything.
# Available extras: dmc, atari, crafter, metaworld, memorymaze, isaaclab, all
pip install -e ".[dmc,metaworld]"
```

Run training on default settings:

```bash
python3 train.py logdir=./logdir/test
```

Training updates `latest.pt` every 1,000,000 environment steps and at the
end of the run. Each save replaces the previous checkpoint, including its replay
data. Set `trainer.save_every=0` to save only at the end or
`trainer.eval_every=0` to skip evaluation. Both intervals use environment steps.

```bash
python3 train.py logdir=./logdir/test trainer.save_every=50000 trainer.eval_every=10000
```

To resume, repeat the original configuration overrides and pass the checkpoint
path. The step budget is the total for the run. The agent, optimizer, and replay
are restored, and the environments start fresh episodes.

```bash
python3 train.py logdir=./logdir/test resume=./logdir/test/latest.pt
```

Monitoring results:
```bash
tensorboard --logdir ./logdir
```

Switching algorithms:

```bash
# Choose an algorithm via model.rep_loss:
# r2dreamer|dreamer|infonce|dreamerpro
python3 train.py model.rep_loss=r2dreamer
```

For easier code reading, inline tensor shape annotations are provided. See [`docs/tensor_shapes.md`](docs/tensor_shapes.md).


## Available Benchmarks
At the moment, the following benchmarks are available in this repository.

| Environment        | Observation | Action | Budget | Description |
|-------------------|---|---|---|-----------------------|
| [Meta-World](https://github.com/Farama-Foundation/Metaworld) | Image | Continuous | 1M | Robotic manipulation with complex contact interactions.|
| [DMC Proprio](https://github.com/deepmind/dm_control) | State | Continuous | 500K | DeepMind Control Suite with low-dimensional inputs. |
| [DMC Vision](https://github.com/deepmind/dm_control) | Image | Continuous |1M| DeepMind Control Suite with high-dimensional images inputs. |
| [DMC Subtle](envs/dmc_subtle.py) | Image | Continuous |1M| DeepMind Control Suite with tiny task-relevant objects. |
| [Atari 100k](https://github.com/Farama-Foundation/Arcade-Learning-Environment) | Image | Discrete |400K| 26 Atari games. |
| [Crafter](https://github.com/danijar/crafter) | Image | Discrete |1M| Survival environment to evaluates diverse agent abilities.|
| [Minecraft Diamond](https://github.com/minerllabs/minerl) | Image and State | Discrete |100M| Collect resources and craft tools to obtain a diamond. |
| [Memory Maze](https://github.com/jurgisp/memory-maze) | Image |Discrete |100M| 3D mazes to evaluate RL agents' long-term memory.|
| [IsaacLab](https://github.com/isaac-sim/IsaacLab) | State / Image | Continuous | 0.5M–1M | GPU-parallelized robotic simulation. |

Use Hydra to select a benchmark and a specific task using `env` and `env.task`, respectively.

```bash
python3 train.py ... env=dmc_vision env.task=dmc_walker_walk
```

## Minecraft Diamond

The patched MineRL wheel targets Linux x86-64 and CPython 3.11. Use a dedicated
environment because MineRL requires Gym 0.23.1. Install Java 8 and make its
`java` executable available on `PATH`. Minecraft also needs an X display.
For an SSH-only machine, follow the [headless Minecraft setup](docs/minecraft.md).

From this repository directory:

```bash
uv venv --python 3.11 ../.venv-minecraft
source ../.venv-minecraft/bin/activate
uv pip install -e . cloudpickle \
  https://github.com/NM512/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
python train.py env=minecraft model.rep_loss=dreamer
```

The preset starts one environment with separate evaluation disabled. Set
`env.env_num=16` to use the legacy PyTorch parallelism setting. For R2-Dreamer,
set `model.rep_loss=r2dreamer`. Startup applies the benchmark rendering settings
to the MineRL installation in the active environment.

## Headless rendering

If you run MuJoCo-based environments (DMC / MetaWorld) on headless machines, you may need to set `MUJOCO_GL` for offscreen rendering. **Using EGL is recommended** as it accelerates rendering, leading to faster simulation throughput.

```bash
# For example, when using EGL (GPU)
export MUJOCO_GL=egl
# (optional) Choose which GPU EGL uses
export MUJOCO_EGL_DEVICE_ID=0
```

More details: [Working with MuJoCo-based environments](https://docs.pytorch.org/rl/stable/reference/generated/knowledge_base/MUJOCO_INSTALLATION.html)

## Code formatting

If you want automatic formatting/basic checks before commits, you can enable `pre-commit`:

```bash
pip install pre-commit
# This sets up a pre-commit hook so that checks are run every time you commit
pre-commit install
# Manual pre-commit run on all files
pre-commit run --all-files
```

## Troubleshooting

If you encounter installation or runtime issues, please refer to our [Troubleshooting Guide](docs/troubleshooting.md).

## Citation

If you find this code useful, please consider citing:

```bibtex
@inproceedings{
morihira2026rdreamer,
title={R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation},
author={Naoki Morihira and Amal Nahar and Kartik Bharadwaj and Yasuhiro Kato and Akinobu Hayashi and Tatsuya Harada},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=Je2QqXrcQq}
}
```

[r2dreamer]: https://openreview.net/forum?id=Je2QqXrcQq&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICLR.cc%2F2026%2FConference%2FAuthors%23your-submissions)
[dreamerv3-torch]: https://github.com/NM512/dreamerv3-torch
