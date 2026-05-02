from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


def _sort_key(name: str) -> List[int]:
    return [int(n) for n in re.findall(r"\d+", name)]


def parse_stroke_txt(path: str) -> List[List[Tuple[float, float, float]]]:
    strokes: List[List[Tuple[float, float, float]]] = []
    cur: List[Tuple[float, float, float]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("STROKE"):
                if cur:
                    strokes.append(cur)
                    cur = []
                continue
            parts = line.split()
            if len(parts) >= 3:
                x, y, t = float(parts[0]), float(parts[1]), float(parts[2])
                cur.append((x, y, t))
            elif len(parts) >= 2:
                x, y = float(parts[0]), float(parts[1])
                cur.append((x, y, 0.0))
    if cur:
        strokes.append(cur)
    return strokes


def strokes_to_features(
    strokes: Sequence[Sequence[Tuple[float, float, float]]],
    *,
    max_dt: float = 0.2,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert strokes to per-point features for RNN.

    Output shape: (T, F) where F=6:
      [x, y, dx, dy, dt, pen_up]
    pen_up is 1.0 on first point of each stroke (stroke boundary), else 0.0.
    """
    pts: List[Tuple[float, float, float, float]] = []
    # store (x,y,t,pen_up)
    for s in strokes:
        if not s:
            continue
        for i, (x, y, t) in enumerate(s):
            pts.append((x, y, t, 1.0 if i == 0 else 0.0))

    if len(pts) < 2:
        return np.zeros((0, 6), dtype=np.float32)

    arr = np.array(pts, dtype=np.float32)  # (T,4)
    x = arr[:, 0]
    y = arr[:, 1]
    t = arr[:, 2]
    pen_up = arr[:, 3]

    # normalize: center + scale to unit box
    x = x - x.mean()
    y = y - y.mean()
    scale = max(float(np.max(np.abs(x))), float(np.max(np.abs(y))), eps)
    x = x / scale
    y = y / scale

    dx = np.zeros_like(x)
    dy = np.zeros_like(y)
    dt = np.zeros_like(t)
    dx[1:] = x[1:] - x[:-1]
    dy[1:] = y[1:] - y[:-1]
    dt[1:] = t[1:] - t[:-1]
    dt = np.clip(dt, 0.0, max_dt)

    feats = np.stack([x, y, dx, dy, dt, pen_up], axis=1).astype(np.float32)
    return feats


@dataclass(frozen=True)
class LabelCodec:
    blank_id: int
    char2id: Dict[str, int]
    id2char: Dict[int, str]

    @property
    def vocab_size(self) -> int:
        return len(self.char2id)

    def encode(self, text: str) -> List[int]:
        return [self.char2id[ch] for ch in text]

    def decode_ids(self, ids: Sequence[int]) -> str:
        return "".join(self.id2char[i] for i in ids)


def build_codec_from_labels_file(labels_path: str) -> LabelCodec:
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f if l.strip()]
    charset = sorted({ch for s in lines for ch in s})
    # CTC: blank reserved at 0, symbols start at 1
    char2id = {ch: i + 1 for i, ch in enumerate(charset)}
    id2char = {i: ch for ch, i in char2id.items()}
    return LabelCodec(blank_id=0, char2id=char2id, id2char=id2char)


class StrokeCtcDataset(Dataset):
    def __init__(self, dataset_root: str, labels_path: str) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.labels_path = labels_path

        with open(labels_path, "r", encoding="utf-8") as f:
            self.label_lines = [l.strip() for l in f if l.strip()]
        self.codec = build_codec_from_labels_file(labels_path)

        self.samples: List[Tuple[str, int]] = []
        # sample: (filepath, line_index)
        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(f"dataset root not found: {dataset_root}")

        for user in sorted(os.listdir(dataset_root)):
            user_path = os.path.join(dataset_root, user)
            if not os.path.isdir(user_path):
                continue
            files = [
                f
                for f in os.listdir(user_path)
                if f.endswith(".txt") and f != "user_info.json"
            ]
            files = sorted(files, key=_sort_key)
            for fname in files:
                m = re.match(r"^(\d+)-\d+\.txt$", fname)
                if not m:
                    continue
                line_1based = int(m.group(1))
                idx0 = line_1based - 1
                if 0 <= idx0 < len(self.label_lines):
                    self.samples.append((os.path.join(user_path, fname), idx0))

        if not self.samples:
            raise RuntimeError(f"no stroke samples found under: {dataset_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        path, idx0 = self.samples[i]
        strokes = parse_stroke_txt(path)
        x = strokes_to_features(strokes)  # (T,F)
        text = self.label_lines[idx0]
        y = np.asarray(self.codec.encode(text), dtype=np.int64)  # (U,)
        return torch.from_numpy(x), torch.from_numpy(y), text


def collate_ctc(batch):
    xs, ys, texts = zip(*batch)
    lengths = torch.tensor([x.shape[0] for x in xs], dtype=torch.long)
    if torch.any(lengths <= 0):
        raise RuntimeError("found empty stroke sample (sequence length 0)")
    max_t = int(torch.max(lengths).item())
    feat_dim = xs[0].shape[1]
    xpad = torch.zeros((len(xs), max_t, feat_dim), dtype=torch.float32)
    for i, x in enumerate(xs):
        t = x.shape[0]
        xpad[i, :t] = x

    y_lens = torch.tensor([y.shape[0] for y in ys], dtype=torch.long)
    ycat = torch.cat(ys, dim=0).to(torch.long)
    return xpad, lengths, ycat, y_lens, texts

