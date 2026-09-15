#!/usr/bin/env python3
"""Profile every section of the training loop."""

import os
import sys
import time
import json
import gc
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import mlx.core as mx
import mlx.optimizers as opt
from mlx.utils import tree_map
from src.model.transformer import init_model, forward, count_parameters
from src.data.dataset import StreamingDataset
from src.training.loss import cross_entropy_loss
from src.model.transformer import causal_mask


def main():
    mx.set_default_device(mx.gpu)

    with open("configs/training_config.json") as f:
        cfg = json.load(f)
    with open("configs/model_config.json") as f:
        model_cfg = json.load(f)

    params, buffers, _ = init_model("configs/model_config.json")
    params = _to_dtype(params, mx.bfloat16)
    mx.eval(params)

    train_paths = cfg["data_paths"]
    train_dataset = StreamingDataset(train_paths, seq_length=cfg["sequence_length"])

    peak_lr = cfg["learning_rate"]
    warmup = opt.linear_schedule(0.0, peak_lr, cfg["warmup_steps"])
    cosine = opt.cosine_decay(peak_lr, cfg["max_steps"] - cfg["warmup_steps"])
    lr_schedule = opt.join_schedules([warmup, cosine], [cfg["warmup_steps"]])
    optimizer = opt.AdamW(learning_rate=lr_schedule, betas=[0.9, 0.999])

    batch_size = cfg["batch_size"]
    grad_accum_steps = cfg.get("gradient_accumulation_steps", 1)
    eval_interval = cfg.get("grad_accum_eval_interval", 2)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    def loss_fn(p, x_batch, y_batch):
        logits = forward(p, buffers, x_batch, model_cfg, training=True)
        return cross_entropy_loss(logits, y_batch)

    grad_fn = mx.compile(mx.value_and_grad(loss_fn))
    batch_iter = train_dataset.iterate_batches(batch_size)

    profile_steps = 50
    timings = {
        "data_load": [],
        "grad_compute": [],
        "tree_ops": [],
        "eval_grads": [],
        "clip_grads": [],
        "optim_step": [],
        "full_step": [],
    }
    micro_timings = {
        "micro_data_load": [],
        "micro_grad": [],
        "micro_tree": [],
        "micro_eval": [],
    }

    print(f"\n{'='*65}")
    print(f"  Profiling: batch={batch_size}, grad_accum={grad_accum_steps}, seq_len={cfg['sequence_length']}")
    print(f"  Params: {count_parameters(params):,}")
    print(f"{'='*65}\n")

    for step in range(profile_steps):
        step_start = time.perf_counter()
        accumulated_grads = None

        loss = mx.array(0.0)

        for mi in range(grad_accum_steps):
            t0 = time.perf_counter()
            x_batch, y_batch = next(batch_iter)
            t1 = time.perf_counter()
            loss, grads = grad_fn(params, x_batch, y_batch)
            t2 = time.perf_counter()
            grads = tree_map(lambda g: g / grad_accum_steps, grads)
            if accumulated_grads is None:
                accumulated_grads = grads
            else:
                accumulated_grads = tree_map(lambda a, g: a + g, accumulated_grads, grads)
            t3 = time.perf_counter()

            if mi % eval_interval == eval_interval - 1 or mi == grad_accum_steps - 1:
                mx.eval(loss, accumulated_grads)
                t4 = time.perf_counter()
            else:
                t4 = t3

            micro_timings["micro_data_load"].append(t1 - t0)
            micro_timings["micro_grad"].append(t2 - t1)
            micro_timings["micro_tree"].append(t3 - t2)
            micro_timings["micro_eval"].append(t4 - t3)

        del grads

        t5 = time.perf_counter()
        if max_grad_norm > 0:
            accumulated_grads, grad_norm = opt.clip_grad_norm(accumulated_grads, max_norm=max_grad_norm)
            mx.eval(accumulated_grads)
        t6 = time.perf_counter()

        optimizer.update(params, accumulated_grads)
        mx.eval(params, optimizer.state)
        t7 = time.perf_counter()

        last_loss = float(loss)
        del accumulated_grads, loss
        gc.collect()
        mx.clear_cache()
        t8 = time.perf_counter()

        timings["data_load"].append(micro_timings["micro_data_load"][-1])
        timings["grad_compute"].append(sum(micro_timings["micro_grad"][-grad_accum_steps:]))
        timings["tree_ops"].append(sum(micro_timings["micro_tree"][-grad_accum_steps:]))
        timings["eval_grads"].append(sum(micro_timings["micro_eval"][-grad_accum_steps:]))
        timings["clip_grads"].append(t6 - t5)
        timings["optim_step"].append(t7 - t6)
        timings["full_step"].append(t8 - step_start)

        print(f"  step {step:>3d} | loss {last_loss:.4f} | "
              f"data={timings['data_load'][-1]*1000:.0f}ms "
              f"fwd_bwd={timings['grad_compute'][-1]*1000:.0f}ms "
              f"tree={timings['tree_ops'][-1]*1000:.0f}ms "
              f"eval={timings['eval_grads'][-1]*1000:.0f}ms "
              f"clip={timings['clip_grads'][-1]*1000:.0f}ms "
              f"optim={timings['optim_step'][-1]*1000:.0f}ms "
              f"total={timings['full_step'][-1]*1000:.0f}ms")

    print(f"\n{'='*65}")
    print(f"  AVERAGE TIMING BREAKDOWN (per step, last {profile_steps} steps)")
    print(f"{'='*65}")

    def avg(lst):
        return sum(lst[-profile_steps:]) / min(len(lst), profile_steps)

    data_avg = avg(timings["data_load"])
    grad_avg = avg(timings["grad_compute"])
    tree_avg = avg(timings["tree_ops"])
    eval_avg = avg(timings["eval_grads"])
    clip_avg = avg(timings["clip_grads"])
    optim_avg = avg(timings["optim_step"])
    total_avg = avg(timings["full_step"])

    labels = [
        ("data_load (next batch)", data_avg),
        ("forward + backward", grad_avg),
        ("tree_map ops", tree_avg),
        ("mx.eval grads", eval_avg),
        ("gradient clipping", clip_avg),
        ("optimizer step", optim_avg),
    ]

    other = total_avg - sum(v for _, v in labels)
    if other > 0:
        labels.append(("cleanup / overhead", other))

    for name, t in labels:
        pct = (t / total_avg) * 100 if total_avg > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:<28s} {t*1000:>8.1f}ms  {pct:>5.1f}%  {bar}")

    print(f"  {'─'*52}")
    print(f"  {'total':<28s} {total_avg*1000:>8.1f}ms  100.0%  ████████████████████████████████████████████")
    print()

    tok_per_step = batch_size * grad_accum_steps * cfg["sequence_length"]
    tok_per_sec = tok_per_step / total_avg if total_avg > 0 else 0
    print(f"  Tokens per step: {tok_per_step:,}")
    print(f"  Tokens/sec: {tok_per_sec:.0f}")
    print(f"  Time per step: {total_avg*1000:.0f} ms")
    print()

    # Micro-batch breakdown
    print(f"{'='*65}")
    print(f"  MICRO-BATCH BREAKDOWN (per micro-step, averaged)")
    print(f"{'='*65}")

    mb_data = avg(micro_timings["micro_data_load"])
    mb_grad = avg(micro_timings["micro_grad"])
    mb_tree = avg(micro_timings["micro_tree"])
    mb_eval = avg(micro_timings["micro_eval"])
    mb_total = mb_data + mb_grad + mb_tree + mb_eval

    for name, t in [("data_load", mb_data), ("grad_fn", mb_grad), ("tree_ops", mb_tree), ("mx.eval", mb_eval)]:
        pct = (t / mb_total) * 100 if mb_total > 0 else 0
        bar = "█" * int(pct / 2)
        print(f"  {name:<20s} {t*1000:>8.2f}ms  {pct:>5.1f}%  {bar}")
    print(f"  {'─'*40}")
    print(f"  {'micro-step total':<20s} {mb_total*1000:>8.2f}ms  100.0%")

    # Key insight: GPU compute vs CPU overhead
    gpu_time = mb_grad + mb_eval
    cpu_time = mb_data + mb_tree
    print(f"\n  GPU compute:  {gpu_time*1000:.1f}ms ({gpu_time/mb_total*100:.0f}% of micro-step)")
    print(f"  CPU overhead: {cpu_time*1000:.1f}ms ({cpu_time/mb_total*100:.0f}% of micro-step)")

    # Memory info
    d = mx.device_info()
    print(f"\n  Device: {d['device_name']}")
    print(f"  GPU memory active: {mx.get_active_memory() / 1024**2:.0f} MB")
    print(f"  GPU memory peak:   {mx.get_peak_memory() / 1024**2:.0f} MB")

    print(f"\n{'='*65}")
    print(f"  PROFILE COMPLETE")
    print(f"{'='*65}")


def _to_dtype(obj, dtype):
    if isinstance(obj, mx.array):
        return obj.astype(dtype)
    elif isinstance(obj, dict):
        return {k: _to_dtype(v, dtype) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_to_dtype(v, dtype) for v in obj]
    return obj


if __name__ == "__main__":
    main()
