import mlx.core as mx
import numpy as np


class StreamingDataset:
    def __init__(self, data: list[int], seq_length: int = 512):
        self.data = data
        self.seq_length = seq_length

    def __len__(self) -> int:
        return (len(self.data) - 1) // self.seq_length

    def get_batch(self, idx: int) -> tuple[mx.array, mx.array]:
        start = idx * self.seq_length
        end = start + self.seq_length + 1
        chunk = self.data[start:end]
        x = mx.array(chunk[:-1], dtype=mx.int32)
        y = mx.array(chunk[1:], dtype=mx.int32)
        return x, y

    def iterate_batches(self, batch_size: int):
        n = len(self)
        indices = np.random.permutation(n)
        for i in range(0, n - batch_size + 1, batch_size):
            batch_indices = indices[i : i + batch_size]
            xs, ys = [], []
            for idx in batch_indices:
                x, y = self.get_batch(int(idx))
                xs.append(x)
                ys.append(y)
            yield mx.stack(xs), mx.stack(ys)
