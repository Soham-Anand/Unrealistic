#!/usr/bin/env python3
"""DPO (Direct Preference Optimization) for Unrealistic, MLX from-scratch style.

Phase 7d: policy = latest checkpoint (7c-final), ref = frozen pre_phase7d
baseline (never the DPO-moving checkpoint). Pairs from build_dpo.py npz.
Loss: -log σ(β·[(πc−πr) − (refc−refr)]) over response-token logps.

Usage:
  python3 scripts/dpo.py --train-config configs/phase7d_dpo_config.json \
      --model-config configs/model_config.json \
      [--ref-checkpoint baselines/pre_phase7d] [--steps 20 (smoke)]
"""

import os, sys, argparse, json, time, math, gc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np
import mlx.core as mx
import mlx.optimizers as opt
from mlx.utils import tree_map
from src.model.transformer import init_model, forward, count_parameters
from src.inference.samples import write_samples
from src.training.trainer import _to_dtype, _find_latest_checkpoint, _step_from_path
from src.training.checkpointing import save_checkpoint, load_checkpoint, prune_checkpoints

SEQ = 512


def _tok_logps(logits_0, y, plen, eos_id):
    """Manual log-softmax gather (no mx.log_softmax in this MLX). Mirrors loss.py."""
    logits_0 = logits_0.astype(mx.float32)
    normed = logits_0 - mx.max(logits_0, axis=-1, keepdims=True)
    tok_lp = (mx.take_along_axis(normed, y[:, None], axis=1).squeeze(-1)
              - mx.logsumexp(normed, axis=-1))
    mask = (mx.arange(y.shape[0]) >= (plen - 1)) & (y != eos_id)
    return mx.sum(mx.where(mask, tok_lp, 0.0))


def seq_logps(params, buffers, model_cfg, ids, prompt_len, eos_id):
    """Sum of response-token log probs for one ids vector (mx arrays inside)."""
    x = mx.array(ids[:-1], dtype=mx.int32)[None, :]
    y = mx.array(ids[1:], dtype=mx.int32)
    logits = forward(params, buffers, x, model_cfg, training=False)
    return _tok_logps(logits[0], y, mx.array(prompt_len, dtype=mx.int32), eos_id)


def logsigmoid(x):
    return -mx.logaddexp(mx.zeros_like(x), -x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-config", default="configs/phase7d_dpo_config.json")
    ap.add_argument("--model-config", default="configs/model_config.json")
    ap.add_argument("--ref-checkpoint", default="baselines/pre_phase7d")
    ap.add_argument("--policy-from", default=None, help="policy ckpt (default: latest in checkpoints/)")
    ap.add_argument("--steps", type=int, default=0, help="smoke override for max_steps delta")
    args = ap.parse_args()

    with open(args.train_config) as f:
        cfg = json.load(f)
    from src.data.tokenizer import Tokenizer
    eos_id = Tokenizer("data/tokenizer/phase1.model").eos_token

    mx.set_default_device(mx.gpu)
    beta = cfg.get("dpo_beta", 0.5)
    lr = cfg.get("learning_rate", 1e-6)
    ga = cfg.get("gradient_accumulation_steps", 4)
    max_steps = cfg["max_steps"]
    save_dir = cfg.get("save_dir", "checkpoints")
    keep_ckpts = cfg.get("keep_last_checkpoints", 3)

    latest = args.policy_from or _find_latest_checkpoint("checkpoints")
    print(f"Policy from: {latest}")
    params, saved_cfg, _ = load_checkpoint(latest, return_opt=True)
    step0 = _step_from_path(latest)
    model_cfg = saved_cfg if saved_cfg else json.load(open(args.model_config))
    _, buffers, _ = init_model(args.model_config)
    dtype = {"float16": mx.float16, "float32": mx.float32,
             "bfloat16": mx.bfloat16}[cfg.get("dtype", "bfloat16")]
    params = _to_dtype(params, dtype)
    mx.eval(params)

    print(f"Ref from: {args.ref_checkpoint}")
    ref_params, _, _ = load_checkpoint(args.ref_checkpoint, return_opt=True)
    ref_params = _to_dtype(ref_params, dtype)
    mx.eval(ref_params)
    print(f"Params: {count_parameters(params):,} | DPO beta={beta} lr={lr} ga={ga}")

    data = np.load(cfg["data_paths"][0])
    C, R, P = data["chosen"], data["rejected"], data["prompt_len"]
    N = int(data["n"][0])
    eval_data = np.load(cfg["eval_path"])
    EC, ER, EP = eval_data["chosen"], eval_data["rejected"], eval_data["prompt_len"]
    EN = int(eval_data["n"][0])
    print(f"Pairs: train {N}, eval {EN}")
    rng = np.random.default_rng(cfg.get("seed", 42))
    order = rng.permutation(N)

    optimizer = opt.AdamW(learning_rate=lr, betas=[0.9, 0.999], eps=1e-8,
                          weight_decay=cfg.get("weight_decay", 0.0))

    def seq_lp_arr(p, ids, plen):
        # ids: (SEQ,) int32 array; plen: scalar int array. Fixed shapes -> compilable.
        y = ids[1:]
        logits = forward(p, buffers, ids[:-1][None, :], model_cfg, training=True)
        return _tok_logps(logits[0], y, plen, eos_id)

    def dpo_loss(p, c, r, plen, rc, rr):
        margin = (seq_lp_arr(p, c, plen) - seq_lp_arr(p, r, plen)) - (rc - rr)
        return -logsigmoid(beta * margin)

    grad_fn = mx.value_and_grad(dpo_loss)
    compiled_grad_fn = mx.compile(grad_fn)

    def to_arr(ids):
        return mx.array(np.asarray(ids, dtype=np.uint16).astype(np.int32))

    def ref_pair(c_ids, r_ids, plen):
        c = to_arr(c_ids)
        r = to_arr(r_ids)
        pl = mx.array(plen, dtype=mx.int32)
        rc = seq_lp_arr(ref_params, c, pl)
        rr = seq_lp_arr(ref_params, r, pl)
        mx.eval(rc, rr)
        return mx.array(float(rc)), mx.array(float(rr))

    def eval_margin(c_ids, r_ids, plen, rc, rr):
        c = to_arr(c_ids)
        r = to_arr(r_ids)
        pl = mx.array(plen, dtype=mx.int32)
        pi_c = seq_lp_arr(params, c, pl)
        pi_r = seq_lp_arr(params, r, pl)
        mx.eval(pi_c, pi_r)
        m = float(pi_c) - float(pi_r) - (float(rc) - float(rr))
        return float(np.logaddexp(0.0, -beta * m)), m

    if args.steps > 0:
        max_steps = step0 + args.steps
        print(f"SMOKE: {step0} -> {max_steps}")

    oi = 0
    for step in range(step0, max_steps):
        t0 = time.perf_counter()
        acc_g = None
        for mi in range(ga):
            if oi >= N:
                oi = 0
                order = rng.permutation(N)
            i = int(order[oi])
            oi += 1
            c_ids = [int(v) for v in C[i]]
            r_ids = [int(v) for v in R[i]]
            plen = int(P[i])
            rc, rr = ref_pair(c_ids, r_ids, plen)
            c = to_arr(c_ids)
            r = to_arr(r_ids)
            pl = mx.array(plen, dtype=mx.int32)
            loss, grads = compiled_grad_fn(params, c, r, pl, rc, rr)
            mx.eval(loss, grads)
            grads = tree_map(lambda g: g / ga, grads)
            acc_g = grads if acc_g is None else tree_map(lambda a, g: a + g, acc_g, grads)
            del grads, c, r, pl
        mx.eval(acc_g)
        if cfg.get("max_grad_norm", 0) > 0:
            acc_g, _ = opt.clip_grad_norm(acc_g, max_norm=cfg["max_grad_norm"])
            mx.eval(acc_g)
        optimizer.update(params, acc_g)
        mx.eval(params, optimizer.state)
        dt = time.perf_counter() - t0
        del acc_g
        gc.collect()
        mx.clear_cache()
        if step % cfg.get("log_every", 5) == 0:
            # cheap margin readout on last micro-batch (no grad)
            l0, m0 = eval_margin(c_ids, r_ids, plen, rc, rr)
            print(f"step {step:>6,} | loss {float(loss):.4f} | margin {m0:+.3f} | "
                  f"{dt:.1f}s/step", flush=True)
        if step % cfg.get("eval_every", 250) == 0 and step > step0:
            tl, ta = 0.0, 0
            for j in range(min(EN, 100)):
                cj = [int(v) for v in EC[j]]
                rj = [int(v) for v in ER[j]]
                pj = int(EP[j])
                rc, rr = ref_pair(cj, rj, pj)
                l_, m_ = eval_margin(cj, rj, pj, rc, rr)
                tl += l_
                ta += 1 if m_ > 0 else 0
            n_ = min(EN, 100)
            print(f"  [eval] step {step} | loss {tl/n_:.4f} | win-rate {ta}/{n_}", flush=True)
        if step % cfg.get("checkpoint_every", 250) == 0 and step > step0:
            save_checkpoint(params, step, save_dir, model_cfg, optimizer.state)
            write_samples(params, buffers, model_cfg,
                          os.path.join(save_dir, f"step_{step:06d}"), step, cfg)
            prune_checkpoints(save_dir, keep_ckpts)

    save_checkpoint(params, step, save_dir, model_cfg, optimizer.state)
    prune_checkpoints(save_dir, keep_ckpts)
    print(f"\nDPO complete at step {step}.")


if __name__ == "__main__":
    main()
