from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import torch


class TokenDataset:
    def __init__(self, data_dir: str | Path, split: str):
        self.data_dir = Path(data_dir)
        metadata = json.loads((self.data_dir / "metadata.json").read_text(encoding="utf-8"))
        self.vocab_size = int(metadata["vocab_size"])
        self.tokens = np.memmap(self.data_dir / f"{split}.bin", dtype=np.uint16, mode="r")

    def batch(
        self,
        batch_size: int,
        seq_len: int,
        device: torch.device,
        generator: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(self.tokens) <= seq_len + 1:
            raise ValueError("token dataset is shorter than one sequence")
        starts = torch.randint(
            0, len(self.tokens) - seq_len - 1, (batch_size,), generator=generator
        ).tolist()
        x = torch.stack(
            [torch.from_numpy(np.asarray(self.tokens[i : i + seq_len], dtype=np.int64)) for i in starts]
        )
        y = torch.stack(
            [torch.from_numpy(np.asarray(self.tokens[i + 1 : i + 1 + seq_len], dtype=np.int64)) for i in starts]
        )
        return x.to(device, non_blocking=True), y.to(device, non_blocking=True)
