import os
import time
import json
import math
import gc
import subprocess
import mlx.core as mx
import mlx.optimizers as opt
from mlx.utils import tree_map
from ..model.transformer import init_model, forward, count_parameters
from ..data.dataset import StreamingDataset
from ..inference.samples import write_samples
from .loss import cross_entropy_loss
from .checkpointing import save_checkpoint, load_checkpoint, prune_checkpoints
from ..utils.memory import get_memory_usage


def train(config_path: str = "configs/model_config.json",
          train_config_path: str = "configs/training_config.json",
          resume: bool = True,
          reset_optimizer: bool = False):
    with open(train_config_path) as f:
        cfg = json.load(f)
    cfg["reset_optimizer"] = reset_optimizer

    mx.set_default_device(mx.gpu)

    save_dir = cfg.get("save_dir", "checkpoints")
    params = buffers = model_cfg = None
    step = 0

    keep_ckpts = cfg.get("keep_last_checkpoints", 3)

    # Resume or init
    if resume:
        latest = _find_latest_checkpoint(save_dir)
        if latest:
            print(f"Resuming from checkpoint: {latest}")
            params, saved_cfg, opt_state = load_checkpoint(latest, return_opt=True)
            step = _step_from_path(latest)
            model_cfg = saved_cfg if saved_cfg else json.load(open(config_path))
            print(f"  Resumed at step {step}")

    if params is None:
        params, buffers, model_cfg = init_model(config_path)
        opt_state = None

    # Convert params to configured dtype
    dtype_map = {"float16": mx.float16, "float32": mx.float32, "bfloat16": mx.bfloat16}
    dtype = dtype_map.get(cfg.get("dtype", "float32"), mx.float32)
    params = _to_dtype(params, dtype)
    mx.eval(params)

    # Precompute buffers if not loaded from checkpoint
    if buffers is None:
        _, buffers, _ = init_model(config_path)

    print(f"\n{'='*60}")
    print(f"  Unrealistic 187M — Training")
    print(f"  Parameters: {count_parameters(params):,}")
    print(f"  Dtype: {cfg.get('dtype', 'float32')}")
    print(f"  Optimizer: AdamW (mlx)")
    print(f"  Micro-batch: {cfg['batch_size']} × {cfg.get('gradient_accumulation_steps', 1)} accum")
    print(f"  Effective batch: {cfg['batch_size'] * cfg.get('gradient_accumulation_steps', 1)}")
    print(f"  Memory: {get_memory_usage()}")
    print(f"{'='*60}\n")

    # Data paths
    train_paths = cfg.get("data_paths", [cfg.get("data_path", "data/phase1/tokens.bin")])
    eval_path = cfg.get("eval_path", train_paths[0])
    missing = [p for p in train_paths if not os.path.exists(p)]
    if missing:
        print(f"ERROR: Training data not found: {missing}")
        return

    train_dataset = StreamingDataset(train_paths, seq_length=cfg["sequence_length"])
    val_dataset = StreamingDataset([eval_path], seq_length=cfg["sequence_length"])

    # LR schedule
    peak_lr = cfg["learning_rate"]
    warmup = opt.linear_schedule(0.0, peak_lr, cfg["warmup_steps"])
    cosine = opt.cosine_decay(peak_lr, cfg["max_steps"] - cfg["warmup_steps"])
    lr_schedule = opt.join_schedules([warmup, cosine], [cfg["warmup_steps"]])

    # AdamW optimizer
    optimizer = opt.AdamW(
        learning_rate=lr_schedule,
        betas=[cfg.get("beta1", 0.9), cfg.get("beta2", 0.999)],
        eps=cfg.get("adam_eps", 1e-8),
        weight_decay=cfg.get("weight_decay", 0.1),
    )

    # Restore optimizer state
    if opt_state is not None and not cfg.get("reset_optimizer", False):
        optimizer.state = opt_state

    batch_size = cfg["batch_size"]
    grad_accum_steps = cfg.get("gradient_accumulation_steps", 1)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)
    max_training_time = cfg.get("max_training_time", 0)
    checkpoint_every_minutes = cfg.get("checkpoint_every_minutes", 60)

    start_time = time.time()
    last_log_time = start_time
    last_log_step = step
    last_checkpoint_time = start_time
    last_thermal_check = start_time
    thermal_check_interval = cfg.get("thermal_check_interval", 30)

    def loss_fn(p, x_batch, y_batch):
        logits = forward(p, buffers, x_batch, model_cfg, training=True)
        return cross_entropy_loss(logits, y_batch)

    grad_fn = mx.value_and_grad(loss_fn)
    compiled_grad_fn = mx.compile(grad_fn)
    batch_iter = train_dataset.iterate_batches(batch_size)

    for step in range(step, cfg["max_steps"]):
        if max_training_time > 0 and time.time() - start_time > max_training_time:
            print(f"\nTraining time limit reached ({max_training_time // 60} minutes). Stopping.")
            break

        # Gradient accumulation over micro-batches (eval every N to balance sync vs memory)
        step_start = time.perf_counter()
        eval_interval = cfg.get("grad_accum_eval_interval", 2)
        accumulated_grads = None
        accum_data_time = 0.0
        accum_fwd_bwd_time = 0.0

        for mi in range(grad_accum_steps):
            t0 = time.perf_counter()
            x_batch, y_batch = next(batch_iter)
            t1 = time.perf_counter()
            loss, grads = compiled_grad_fn(params, x_batch, y_batch)
            t2 = time.perf_counter()
            grads = tree_map(lambda g: g / grad_accum_steps, grads)

            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, g: a + g, accumulated_grads, grads)

            if mi % eval_interval == eval_interval - 1 or mi == grad_accum_steps - 1:
                mx.eval(loss, accumulated_grads)
            t3 = time.perf_counter()
            accum_data_time += t1 - t0
            accum_fwd_bwd_time += t3 - t2

        del grads

        # Gradient clipping on accumulated grads
        clip_start = time.perf_counter()
        if max_grad_norm > 0:
            accumulated_grads, grad_norm = opt.clip_grad_norm(accumulated_grads, max_norm=max_grad_norm)
            mx.eval(accumulated_grads)
        clip_time = time.perf_counter() - clip_start

        # Optimizer step — modifies params in-place (MLX functional style)
        opt_start = time.perf_counter()
        optimizer.update(params, accumulated_grads)
        mx.eval(params, optimizer.state)
        opt_time = time.perf_counter() - opt_start

        step_time = time.perf_counter() - step_start

        # Thermal check — cooldown if system is hot
        if time.time() - last_thermal_check >= thermal_check_interval:
            last_thermal_check = time.time()
            _maybe_cooldown()

        # Logging
        if step % cfg.get("log_every", 1) == 0:
            now = time.time()
            elapsed = now - last_log_time
            steps_done = step - last_log_step
            tokens_per_step = batch_size * grad_accum_steps * train_dataset.seq_length
            tok_per_sec = steps_done * tokens_per_step / elapsed if elapsed > 0 else 0
            last_log_time = now
            last_log_step = step
            lr_val = float(optimizer.learning_rate)
            mem_active = mx.get_active_memory() / 1024**2
            mem_peak = mx.get_peak_memory() / 1024**2
            print(f"step {step:>6,} | loss {float(loss):.4f} | "
                  f"lr {lr_val:.2e} | {tok_per_sec:.0f} tok/s | "
                  f"fwd_bwd {accum_fwd_bwd_time*1000:.0f}ms "
                  f"data {accum_data_time*1000:.0f}ms "
                  f"clip {clip_time*1000:.0f}ms "
                  f"optim {opt_time*1000:.0f}ms "
                  f"step {step_time*1000:.0f}ms "
                  f"mem {mem_active:.0f}/{mem_peak:.0f}MB")

        del accumulated_grads, loss
        gc.collect()
        mx.clear_cache()

        # Evaluation
        if step % cfg.get("eval_every", 500) == 0:
            _evaluate(params, buffers, model_cfg, val_dataset, cfg, step)

        # Step-based checkpoint
        if step % cfg.get("checkpoint_every", 1000) == 0:
            save_checkpoint(params, step, cfg["save_dir"], model_cfg, optimizer.state)
            write_samples(params, buffers, model_cfg, os.path.join(cfg["save_dir"], f"step_{step:06d}"), step, cfg)
            prune_checkpoints(cfg["save_dir"], keep_ckpts)

        # Time-based checkpoint
        elapsed_since_ckpt = time.time() - last_checkpoint_time
        if elapsed_since_ckpt >= checkpoint_every_minutes * 60:
            save_checkpoint(params, step, cfg["save_dir"], model_cfg, optimizer.state)
            prune_checkpoints(cfg["save_dir"], keep_ckpts)
            last_checkpoint_time = time.time()

    save_checkpoint(params, step, cfg["save_dir"], model_cfg, optimizer.state)
    prune_checkpoints(cfg["save_dir"], keep_ckpts)
    print(f"\nTraining complete at step {step}.")


def _evaluate(params, buffers, model_cfg, val_dataset, cfg, step):
    total_loss = 0.0
    n_batches = 0
    eval_steps = cfg.get("eval_steps", 50)
    eval_batch_size = cfg.get("eval_batch_size", 1)

    compiled_forward = mx.compile(
        lambda p, b, x: forward(p, b, x, model_cfg, training=False)
    )

    for x, y in val_dataset.iterate_batches(eval_batch_size):
        if n_batches >= eval_steps:
            break
        logits = compiled_forward(params, buffers, x)
        loss = cross_entropy_loss(logits, y)
        total_loss += float(loss)
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    ppl = math.exp(avg_loss)
    print(f"  [eval] step {step} | val_loss {avg_loss:.4f} | ppl {ppl:.2f}")


def _to_dtype(obj, dtype):
    if isinstance(obj, mx.array):
        return obj.astype(dtype)
    elif isinstance(obj, dict):
        return {k: _to_dtype(v, dtype) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dtype(v, dtype) for v in obj]
    return obj


def _find_latest_checkpoint(save_dir: str) -> str | None:
    if not os.path.exists(save_dir):
        return None
    dirs = []
    for d in os.listdir(save_dir):
        if d.startswith("step_") and os.path.isdir(os.path.join(save_dir, d)):
            try:
                step_num = int(d.split("_")[1])
                dirs.append((step_num, os.path.join(save_dir, d)))
            except ValueError:
                continue
    if not dirs:
        return None
    dirs.sort(key=lambda x: x[0])
    return dirs[-1][1]


def _step_from_path(path: str) -> int:
    name = os.path.basename(path)
    return int(name.split("_")[1])


def _maybe_cooldown(cooldown_seconds: int = 30):
    """Check pmset thermal status and pause if system is overheating."""
    try:
        result = subprocess.run(
            ["pmset", "-g", "therm"],
            capture_output=True, text=True, timeout=5,
        )
        if "Warning" in result.stdout:
            print(f"  [thermal] System under pressure, cooling {cooldown_seconds}s...")
            time.sleep(cooldown_seconds)
    except Exception:
        pass
