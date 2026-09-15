import json
import os
import shutil
import re
import mlx.core as mx


def prune_checkpoints(save_dir: str, keep: int = 3):
    """Delete old checkpoints, keeping only the newest `keep`. No-op if keep <= 0."""
    if keep is None or keep <= 0:
        return
    if not os.path.isdir(save_dir):
        return

    dirs = []
    for name in os.listdir(save_dir):
        m = re.match(r"^step_(\d+)$", name)
        if m and os.path.isdir(os.path.join(save_dir, name)):
            dirs.append((int(m.group(1)), name))

    dirs.sort(reverse=True)
    for step_num, name in dirs[keep:]:
        path = os.path.join(save_dir, name)
        print(f"  Pruning old checkpoint: {path}")
        shutil.rmtree(path, ignore_errors=True)


def save_checkpoint(params: dict, step: int, save_dir: str = "checkpoints",
                    model_cfg: dict = None, opt_state: dict = None):
    """Save model parameters, optimizer state, and config."""
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{step:06d}")
    os.makedirs(path, exist_ok=True)

    flat = {}
    _flatten(params, prefix="params", flat=flat)

    if opt_state:
        _flatten(opt_state, prefix="opt", flat=flat)

    mx.savez(os.path.join(path, "model.npz"), **flat)

    meta = {"step": step}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f)

    if model_cfg is not None:
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(model_cfg, f, indent=2)

    print(f"Checkpoint saved: {path}")


def load_checkpoint(checkpoint_dir: str, return_opt: bool = False):
    """Load model parameters and optionally optimizer state."""
    model_path = os.path.join(checkpoint_dir, "model.npz")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No checkpoint found at {model_path}")

    flat = mx.load(model_path)

    param_items = {k: v for k, v in flat.items() if k.startswith("params.")}
    params = _unflatten(param_items, strip_prefix="params")

    config_path = os.path.join(checkpoint_dir, "config.json")
    cfg = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)

    opt_state = None
    if return_opt:
        opt_items = {k: v for k, v in flat.items() if k.startswith("opt.")}
        if opt_items:
            opt_state = _unflatten(opt_items, strip_prefix="opt")

    if return_opt:
        return params, cfg, opt_state
    return params, cfg


def _flatten(d, prefix="", flat=None):
    if flat is None:
        flat = {}
    for k, v in d.items():
        full = f"{prefix}.{k}"
        if isinstance(v, mx.array):
            flat[full] = v
        elif isinstance(v, dict):
            _flatten(v, full, flat=flat)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                _flatten(item, f"{full}.{i}", flat=flat)
    return flat


def _unflatten(flat: dict, strip_prefix: str = "") -> dict:
    result = {}
    for key, value in flat.items():
        if strip_prefix:
            key = key[len(strip_prefix) + 1:]
        parts = key.split(".")
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return _convert_lists(result)


def _convert_lists(d):
    if isinstance(d, dict):
        if d and all(k.isdigit() for k in d.keys()):
            return [_convert_lists(d[str(i)]) for i in range(len(d))]
        return {k: _convert_lists(v) for k, v in d.items()}
    return d
