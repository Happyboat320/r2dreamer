from importlib.util import find_spec
from pathlib import Path


def configure_minecraft():
    # All MineRL workers share this installation. Apply options before spawning them.
    spec = find_spec("minerl")
    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Minecraft requires the patched MineRL wheel. See the Minecraft section in README.md.")

    options = Path(spec.origin).parent / "Malmo" / "Minecraft" / "run" / "options.txt"
    settings = {
        "difficulty": "2",
        "renderDistance": "6",
        "simulationDistance": "6",
        "fovEffectScale": "0.0",
        "ao": "1",
        "gamma": "5.0",
    }
    original = options.read_text()
    lines = [line for line in original.splitlines() if line.partition(":")[0] not in settings]
    lines.extend(f"{key}:{value}" for key, value in settings.items())
    updated = "\n".join(lines) + "\n"
    if updated != original:
        options.write_text(updated)
