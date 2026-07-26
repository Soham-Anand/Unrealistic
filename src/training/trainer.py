import time
import mlx.core as mx
from ..model.transformer import UnrealisticModel
from ..data.dataset import StreamingDataset
from .loss import cross_entropy_loss
from .scheduler import CosineScheduler
from .checkpointing import save_checkpoint
from ..utils.metrics import compute_perplexity
from ..utils.memory import get_memory_usage


class Trainer:
    def __init__(self, model: UnrealisticModel, config: dict):
        self.model = model
        self.config = config
        self.scheduler = CosineScheduler(
            max_steps=config["max_steps"],
            warmup_steps=config["warmup_steps"],
            peak_lr=config["learning_rate"],
        )
        self.step = 0
        self._grad_accum = config.get("gradient_accumulation_steps", 1)

    def train_step(self, x: mx.array, y: mx.array) -> float:
        def loss_fn(model):
            logits = model(x)
            return cross_entropy_loss(logits, y) / self._grad_accum

        loss, grads = mx.value_and_grad(self.model)(loss_fn)

        if not hasattr(self, "_accumulated_grads"):
            self._accumulated_grads = grads
        else:
            self._accumulated_grads = _add_grads(self._accumulated_grads, grads)

        self._grad_count = getattr(self, "_grad_count", 0) + 1

        if self._grad_count >= self._grad_accum:
            lr = self.scheduler(self.step)
            _apply_grads(self.model, self._accumulated_grads, lr, self.config["weight_decay"])
            self._accumulated_grads = None
            self._grad_count = 0
            self.step += 1

        return float(loss)

    def train(self, dataset: StreamingDataset):
        batch_size = self.config["batch_size"]
        print(f"\n{'='*60}")
        print(f"  Unrealistic 187M — Training")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Memory: {get_memory_usage()}")
        print(f"  Steps: {self.config['max_steps']:,}")
        print(f"{'='*60}\n")

        start_time = time.time()
        losses = []

        for x, y in dataset.iterate_batches(batch_size):
            if self.step >= self.config["max_steps"]:
                break

            loss = self.train_step(x, y)
            losses.append(loss)

            if self.step % self.config.get("log_every", 10) == 0 and self.step > 0:
                avg_loss = sum(losses[-100:]) / len(losses[-100:])
                elapsed = time.time() - start_time
                tok_per_sec = self.step * batch_size * dataset.seq_length / elapsed
                print(
                    f"step {self.step:>6,} | loss {avg_loss:.4f} | "
                    f"lr {self.scheduler(self.step):.2e} | "
                    f"{tok_per_sec:.0f} tok/s"
                )

            if self.step > 0 and self.step % self.config.get("checkpoint_every", 1000) == 0:
                save_checkpoint(self.model, None, self.step, self.config["save_dir"])

        save_checkpoint(self.model, None, self.step, self.config["save_dir"])
        print(f"\nTraining complete. Final loss: {losses[-1]:.4f}")


def _add_grads(g1, g2):
    if g1 is None:
        return g2
    if isinstance(g1, mx.array):
        return g1 + g2
    return {k: _add_grads(g1[k], g2[k]) for k in g1}


def _apply_grads(model, grads, lr, weight_decay):
    params = [model.embeddings.token_embedding]
    grad_list = [grads["embeddings"]]

    for i, layer in enumerate(model.layers):
        prefix = f"layer_{i}"
        params.extend([
            layer.attention_norm.weight,
            layer.attention.Wq, layer.attention.Wk,
            layer.attention.Wv, layer.attention.Wo,
            layer.ffn_norm.weight,
            layer.ffn.w_gate, layer.ffn.w_up, layer.ffn.w_down,
        ])
        grad_list.extend([
            grads[f"{prefix}.attn_norm"],
            grads[f"{prefix}.Wq"], grads[f"{prefix}.Wk"],
            grads[f"{prefix}.Wv"], grads[f"{prefix}.Wo"],
            grads[f"{prefix}.ffn_norm"],
            grads[f"{prefix}.w_gate"], grads[f"{prefix}.w_up"], grads[f"{prefix}.w_down"],
        ])

    params.append(model.final_norm.weight)
    grad_list.append(grads["final_norm"])

    for p, g in zip(params, grad_list):
        p -= lr * g
