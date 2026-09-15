import os
import fcntl
import mlx.core as mx
import numpy as np

CHUNK_WORDS = 5_000_000
EOS_ID = 2


class StreamingDataset:
    def __init__(self, paths: list[str], seq_length: int = 512,
                 shuffle_slices: bool = True):
        self.paths = paths if isinstance(paths, list) else [paths]
        self.seq_length = seq_length
        self.shuffle_slices = shuffle_slices
        self.lengths = [os.path.getsize(p) // 2 for p in self.paths]

    def __len__(self) -> int:
        total = sum(l for l in self.lengths)
        return (total - 1) // self.seq_length

    def _read(self, path: str, pos: int, n: int) -> np.ndarray:
        total = os.path.getsize(path) // 2
        n = min(n, total - pos)
        if n <= 0:
            return np.array([], dtype=np.uint16)
        fd = os.open(path, os.O_RDONLY)
        fcntl.fcntl(fd, fcntl.F_NOCACHE, 1)
        with os.fdopen(fd, "rb") as f:
            f.seek(pos * 2)
            raw = f.read(n * 2)
        return np.frombuffer(raw, dtype=np.uint16)

    def iterate_batches(self, batch_size: int):
        order = list(range(len(self.paths)))
        if self.shuffle_slices:
            rng = np.random.default_rng()
            rng.shuffle(order)

        while True:
            for si in order:
                path = self.paths[si]
                total = self.lengths[si]
                if total <= self.seq_length + 1:
                    continue

                buf_start = 0
                buf = self._read(path, 0, CHUNK_WORDS)
                n_batches = (total - 1) // self.seq_length
                i = np.random.randint(0, min(CHUNK_WORDS // self.seq_length, n_batches))

                while i < n_batches:
                    need_start = i * self.seq_length
                    while need_start >= buf_start + len(buf) - self.seq_length - 1:
                        buf_start = max(0, need_start)
                        remaining = total - buf_start
                        if remaining <= self.seq_length + 1:
                            break
                        chunk_len = min(CHUNK_WORDS, remaining)
                        buf = self._read(path, buf_start, chunk_len)

                    xs, ys = [], []
                    for b in range(batch_size):
                        if i + b >= n_batches:
                            break
                        start = (i + b) * self.seq_length - buf_start
                        seg = buf[start:start + self.seq_length + 1]
                        if len(seg) < self.seq_length + 1:
                            break
                        xs.append(mx.array(seg[:-1].copy(), dtype=mx.int32))
                        ys.append(mx.array(seg[1:].copy(), dtype=mx.int32))

                    if not xs:
                        i += batch_size
                        continue

                    yield mx.stack(xs), mx.stack(ys)
                    i += batch_size
