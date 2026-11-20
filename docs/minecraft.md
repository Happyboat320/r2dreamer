# Minecraft Diamond on a headless server

These commands target Ubuntu 24.04 on x86-64 with an NVIDIA GPU and a driver
that supports the repository's PyTorch CUDA build. Xvfb supplies an X display,
Mesa renders Minecraft on the CPU, and PyTorch runs the learner on CUDA.
Use the same setup on a workstation to reproduce the server launch.

## Install dependencies

Install the system packages on the server. OpenJDK 8 and Xvfb are available
from Ubuntu's Universe repository.

```bash
sudo apt-get update
sudo apt-get install -y openjdk-8-jre-headless build-essential xvfb xauth \
  libgl1 libglx-mesa0 libgl1-mesa-dri libxrandr2 libxcursor1 libxi6 libxxf86vm1
```

With [uv](https://docs.astral.sh/uv/) installed, run from the directory
containing `train.py`:

```bash
uv venv --python 3.11 ../.venv-minecraft
source ../.venv-minecraft/bin/activate
uv pip install --index-url https://mirrors.ustc.edu.cn/pypi/simple -e . cloudpickle \
  https://github.com/NM512/minerl/releases/download/v0.4.4-patched/minerl_mirror-0.4.4-cp311-cp311-linux_x86_64.whl
```

If the server cannot reach GitHub Releases, copy the wheel to the server and
pass its local path in place of the URL.

## Start training

Select Java 8 in the training shell. For a Java 8 installation elsewhere, set
`JAVA_HOME` to the directory containing `bin/java`.

```bash
source ../.venv-minecraft/bin/activate
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
export PATH="$JAVA_HOME/bin:$PATH"

JAVA_TOOL_OPTIONS="${JAVA_TOOL_OPTIONS:+$JAVA_TOOL_OPTIONS }-Djavax.accessibility.assistive_technologies=" \
  LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a -s '-screen 0 1024x768x24' \
  python train.py env=minecraft model=size12M model.rep_loss=dreamer \
  device=cuda:0 logdir=./logdir/minecraft
```

The JVM option disables the desktop accessibility bridge for this run. Each
environment starts a JVM with a 2 GiB maximum heap. Increase `env.env_num`
after checking host CPU and RAM usage. Replay uses the selected CUDA device by
default. Set `buffer.storage_device=cpu` to store it in host RAM, or lower
`buffer.max_size` to reduce the allocation.

For a short smoke run, append these overrides to the training command:

```text
env.steps=12 batch_size=2 batch_length=4 buffer.max_size=128
model.compile=false trainer.train_ratio=8 trainer.update_log_every=1
```

The log directory contains `console.log`, TensorBoard events, and `latest.pt`
after the run finishes. If Minecraft reports an LWJGL or OpenGL initialization
error, check the Xvfb and Mesa setup on the same machine. With the optional
`mesa-utils` package installed, this command reports the renderer:

```bash
LIBGL_ALWAYS_SOFTWARE=1 xvfb-run -a glxinfo -B
```
