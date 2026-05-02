from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from model.data import StrokeCtcDataset, collate_ctc
from model.metrics import cer, ctc_greedy_decode, wer
from model.scripts.bilstm import BiLstmCtc
from model.scripts.rnn import RnnCtc

_RUNS_ROOT = os.path.join("model", "runs")
_CHECKPOINT_ROOT = os.path.join("model", "checkpoints")


@dataclass(frozen=True)
class RunConfig:
    mode: str
    model: str
    data: str
    labels: str
    split_ratio: float
    hidden: int
    num_layers: int
    dropout: float
    epochs: int | None
    batch_size: int | None
    lr: float | None
    seed: int
    checkpoint_path: str | None
    plot: str
    device: str


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_model(name: str, *, input_size: int, vocab_size: int, cfg: RunConfig) -> nn.Module:
    if name == "rnn":
        return RnnCtc(
            input_size=input_size,
            vocab_size=vocab_size,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    if name == "bilstm":
        return BiLstmCtc(
            input_size=input_size,
            vocab_size=vocab_size,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        )
    raise ValueError(f"unknown --model: {name}")


def split_indices(n: int, split_ratio: float, seed: int) -> Tuple[List[int], List[int]]:
    idx = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(idx)
    n_val = int(round(n * split_ratio))
    n_val = max(1, min(n - 1, n_val))
    val = idx[:n_val]
    tr = idx[n_val:]
    return tr, val


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    blank_id: int,
    codec,
) -> Dict[str, float]:
    model.eval()
    losses: List[float] = []
    cers: List[float] = []
    wers: List[float] = []
    exact = 0
    total = 0

    ctc_loss = nn.CTCLoss(blank=blank_id, zero_infinity=True)

    for xpad, xlen, ycat, ylen, texts in loader:
        xpad = xpad.to(device)
        xlen = xlen.to(device)
        ycat = ycat.to(device)
        ylen = ylen.to(device)

        logp = model(xpad, xlen)  # (T,B,C)
        loss = ctc_loss(logp, ycat, xlen, ylen)
        losses.append(float(loss.item()))

        pred_ids = ctc_greedy_decode(logp, xlen, blank_id=blank_id)
        for hyp_ids, ref in zip(pred_ids, texts):
            hyp = codec.decode_ids(hyp_ids)
            cers.append(cer(ref, hyp))
            wers.append(wer(ref, hyp))
            exact += 1 if hyp == ref else 0
            total += 1

    return {
        "loss": float(np.mean(losses)) if losses else float("nan"),
        "cer": float(np.mean(cers)) if cers else float("nan"),
        "wer": float(np.mean(wers)) if wers else float("nan"),
        "exact": float(exact / max(1, total)),
    }


def train_one_run(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ds = StrokeCtcDataset(cfg.data, cfg.labels)
    input_size = 6
    vocab_size = ds.codec.vocab_size

    tr_idx, va_idx = split_indices(len(ds), cfg.split_ratio, cfg.seed)
    tr_ds = Subset(ds, tr_idx)
    va_ds = Subset(ds, va_idx)

    tr_loader = DataLoader(
        tr_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_ctc,
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_ctc,
    )

    model = make_model(cfg.model, input_size=input_size, vocab_size=vocab_size, cfg=cfg)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    ctc_loss = nn.CTCLoss(blank=ds.codec.blank_id, zero_infinity=True)

    run_ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = os.path.join(_RUNS_ROOT, run_ts)
    os.makedirs(run_dir, exist_ok=True)

    if cfg.checkpoint_path is None:
        os.makedirs(_CHECKPOINT_ROOT, exist_ok=True)
        ckpt_path = os.path.join(
            _CHECKPOINT_ROOT, f"{cfg.model}_seed{cfg.seed}_{run_ts}.pt"
        )
    else:
        ckpt_path = cfg.checkpoint_path

    cfg_record = {**asdict(cfg), "checkpoint_path": ckpt_path}
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg_record, f, ensure_ascii=False, indent=2)

    if ckpt_path:
        ckpt_parent = os.path.dirname(os.path.abspath(ckpt_path))
        if ckpt_parent:
            os.makedirs(ckpt_parent, exist_ok=True)

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])

    history: Dict[str, List[float]] = {"tr_loss": [], "va_loss": [], "va_cer": [], "va_wer": []}

    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_losses: List[float] = []
        for xpad, xlen, ycat, ylen, _texts in tr_loader:
            xpad = xpad.to(device)
            xlen = xlen.to(device)
            ycat = ycat.to(device)
            ylen = ylen.to(device)

            logp = model(xpad, xlen)
            loss = ctc_loss(logp, ycat, xlen, ylen)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            ep_losses.append(float(loss.item()))

        tr_loss = float(np.mean(ep_losses)) if ep_losses else float("nan")
        va = evaluate(model, va_loader, device, ds.codec.blank_id, ds.codec)
        history["tr_loss"].append(tr_loss)
        history["va_loss"].append(va["loss"])
        history["va_cer"].append(va["cer"])
        history["va_wer"].append(va["wer"])

        print(
            f"epoch {ep}/{cfg.epochs}  "
            f"tr_loss={tr_loss:.4f}  va_loss={va['loss']:.4f}  "
            f"va_cer={va['cer']:.4f}  va_wer={va['wer']:.4f}  exact={va['exact']:.3f}"
        )

        if ckpt_path:
            torch.save(
                {"model": model.state_dict(), "opt": opt.state_dict(), "config": cfg_record},
                ckpt_path,
            )

    with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    if cfg.plot == "yes":
        plot_history(history, out_dir=run_dir)


def eval_only(cfg: RunConfig) -> None:
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ds = StrokeCtcDataset(cfg.data, cfg.labels)
    input_size = 6
    vocab_size = ds.codec.vocab_size

    _tr_idx, va_idx = split_indices(len(ds), cfg.split_ratio, cfg.seed)
    va_ds = Subset(ds, va_idx)
    va_loader = DataLoader(
        va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0, collate_fn=collate_ctc
    )

    model = make_model(cfg.model, input_size=input_size, vocab_size=vocab_size, cfg=cfg)
    model.to(device)

    if not cfg.checkpoint_path or not os.path.exists(cfg.checkpoint_path):
        raise FileNotFoundError("eval needs --checkpoint_path pointing to existing checkpoint")
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model"])

    va = evaluate(model, va_loader, device, ds.codec.blank_id, ds.codec)
    print(
        f"eval  loss={va['loss']:.4f}  cer={va['cer']:.4f}  wer={va['wer']:.4f}  exact={va['exact']:.3f}"
    )


def plot_history(history: Dict[str, List[float]], out_dir: str) -> None:
    import matplotlib.pyplot as plt

    xs = list(range(1, len(history["tr_loss"]) + 1))

    plt.figure(figsize=(8, 4))
    plt.plot(xs, history["tr_loss"], label="train loss")
    plt.plot(xs, history["va_loss"], label="val loss")
    plt.xlabel("epoch")
    plt.ylabel("ctc loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "loss.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4))
    plt.plot(xs, history["va_cer"], label="val CER")
    plt.plot(xs, history["va_wer"], label="val WER")
    plt.xlabel("epoch")
    plt.ylabel("error rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_rates.png"), dpi=160)
    plt.close()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Online handwriting recognition (PyTorch, CTC)")

    p.add_argument("--data", required=True, help="dataset root path (contains user folders)")
    p.add_argument("--plot", choices=["yes", "no"], default="no")
    p.add_argument("--labels", default="syl.txt", help="label lines file (utf-8)")

    p.add_argument("--mode", required=True, choices=["train", "eval"])
    p.add_argument("--model", required=True, choices=["rnn", "bilstm"])
    p.add_argument("--split_ratio", required=True, type=float)

    p.add_argument("--hidden", required=True, type=int)
    p.add_argument("--num_layers", required=True, type=int)
    p.add_argument("--dropout", required=True, type=float)

    p.add_argument("--epochs", required=True, type=int)
    p.add_argument("--batch_size", required=True, type=int)
    p.add_argument("--lr", required=True, type=float)

    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--checkpoint_path",
        default=None,
        metavar="PATH",
        help="checkpoint .pt (train default: model/checkpoints/<model>_seed<seed>_<timestamp>.pt, same ts as run dir; eval: required)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    return p


def main() -> None:
    args = build_arg_parser().parse_args()

    if not (0.0 < args.split_ratio < 1.0):
        raise SystemExit("--split_ratio must be in (0,1)")
    if args.hidden <= 0 or args.num_layers <= 0:
        raise SystemExit("--hidden and --num_layers must be > 0")
    if not (0.0 <= args.dropout < 1.0):
        raise SystemExit("--dropout must be in [0,1)")

    cfg = RunConfig(
        mode=args.mode,
        model=args.model,
        data=args.data,
        labels=args.labels,
        split_ratio=args.split_ratio,
        hidden=args.hidden,
        num_layers=args.num_layers,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        checkpoint_path=args.checkpoint_path or None,
        plot=args.plot,
        device=args.device,
    )

    if cfg.mode == "train":
        train_one_run(cfg)
    else:
        eval_only(cfg)


if __name__ == "__main__":
    main()

