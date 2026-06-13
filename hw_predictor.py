## Myanmar Syllable Handwriting Predictor
## Draw a syllable for the model to predict which one
## LOCAL: python hw_predictor.py --file syl.txt --checkpoint_rnn path/to/rnn.pt --checkpoint_bilstm path/to/bilstm.pt
## HF:   python hw_predictor.py --file syl.txt --checkpoint_bilstm hf://user/repo/bilstm.pt

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import torch
from huggingface_hub import hf_hub_download
from PyQt5.QtCore import Qt, QThread, QTimer, pyqtSignal
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)

from hw_collector import DrawingWidget
from main import make_model
from model.data import build_codec_from_labels_file, strokes_to_features
from model.metrics import ctc_beam_decode_topk

INPUT_SIZE = 6
WINDOW_WIDTH = 700
WINDOW_HEIGHT = 820
TOP_K = 5


def resolve_checkpoint(path_or_hf: str, *, hf_token: str | None = None) -> str:
    if os.path.isfile(path_or_hf):
        return path_or_hf
    if path_or_hf.startswith("hf://"):
        rest = path_or_hf[5:]
        repo_id, _, filename = rest.rpartition("/")
        if not repo_id or not filename:
            raise ValueError(
                f"invalid hf checkpoint path: {path_or_hf!r} "
                "(expected hf://user/repo/filename.pt)"
            )
        return hf_hub_download(repo_id=repo_id, filename=filename, token=hf_token)
    raise FileNotFoundError(f"checkpoint not found: {path_or_hf}")


class StrokePredictor:
    def __init__(
        self,
        labels_path: str,
        checkpoint_path: str,
        device: str,
        *,
        hf_token: str | None = None,
    ) -> None:
        self.labels_path = labels_path
        self.checkpoint_path = checkpoint_path
        self.hf_token = hf_token
        self.device = torch.device(device)
        self.codec = build_codec_from_labels_file(labels_path)
        self.local_checkpoint_path = ""
        self._load(checkpoint_path)

    def _load(self, checkpoint_path: str) -> None:
        local_path = resolve_checkpoint(checkpoint_path, hf_token=self.hf_token)
        self.local_checkpoint_path = local_path
        ckpt = torch.load(local_path, map_location="cpu")
        if "model" not in ckpt or "config" not in ckpt:
            raise ValueError(f"invalid checkpoint: {checkpoint_path}")

        cfg_dict = ckpt["config"]
        cfg = SimpleNamespace(
            model=cfg_dict["model"],
            hidden=cfg_dict["hidden"],
            num_layers=cfg_dict["num_layers"],
            dropout=cfg_dict["dropout"],
        )

        self.model = make_model(
            cfg.model,
            input_size=INPUT_SIZE,
            vocab_size=self.codec.vocab_size,
            cfg=cfg,  # type: ignore[arg-type]
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.to(self.device)
        self.model.eval()
        self.model_name = cfg.model
        self.checkpoint_path = checkpoint_path

    def predict_topk(self, strokes, *, top_k: int = TOP_K) -> list[tuple[str, float]]:
        feats = strokes_to_features(strokes)
        if feats.shape[0] == 0:
            raise ValueError("empty drawing")

        x = torch.from_numpy(feats).unsqueeze(0).to(self.device)
        xlen = torch.tensor([feats.shape[0]], dtype=torch.long, device=self.device)

        with torch.no_grad():
            logp = self.model(x, xlen)

        decoded = ctc_beam_decode_topk(
            logp,
            xlen,
            blank_id=self.codec.blank_id,
            beam_size=max(10, top_k * 2),
            top_k=top_k,
        )[0]
        return [
            (self.codec.decode_ids(ids), prob)
            for ids, prob in decoded
        ]

    def predict(self, strokes) -> str:
        topk = self.predict_topk(strokes, top_k=1)
        return topk[0][0] if topk else ""


class ModelLoadWorker(QThread):
    finished = pyqtSignal(str, object)
    failed = pyqtSignal(str, str)

    def __init__(
        self,
        name: str,
        labels_path: str,
        checkpoint_path: str,
        device: str,
        *,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        self.name = name
        self.labels_path = labels_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.hf_token = hf_token

    def run(self) -> None:
        try:
            predictor = StrokePredictor(
                self.labels_path,
                self.checkpoint_path,
                self.device,
                hf_token=self.hf_token,
            )
            self.finished.emit(self.name, predictor)
        except Exception as e:
            self.failed.emit(self.name, str(e))


class PredictorWindow(QWidget):
    def __init__(
        self,
        labels_path: str,
        checkpoints: dict[str, str],
        default_model: str,
        font_size: int,
        device: str,
        *,
        hf_token: str | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Myanmar Syllable Handwriting Predictor")

        self.labels_path = labels_path
        self.checkpoints = checkpoints
        self.device = device
        self.hf_token = hf_token
        self.predictors: dict[str, StrokePredictor] = {}
        self._loading: set[str] = set()
        self._workers: dict[str, ModelLoadWorker] = {}

        self.canvas = DrawingWidget()

        self.model_combo = QComboBox()
        for name in sorted(checkpoints):
            self.model_combo.addItem(name)
        if default_model in checkpoints:
            self.model_combo.setCurrentText(default_model)

        self.prediction_label = QLabel("—")
        self.prediction_label.setAlignment(Qt.AlignCenter)
        self.prediction_label.setFont(QFont("Noto Sans Myanmar", font_size))

        self.topk_label = QLabel("")
        self.topk_label.setAlignment(Qt.AlignCenter)
        self.topk_label.setWordWrap(True)
        self.topk_label.setMinimumHeight(28)

        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignCenter)

        self.model_status = QLabel("")
        self.model_status.setAlignment(Qt.AlignCenter)
        self.model_status.setWordWrap(True)
        self.model_status.setMaximumHeight(40)

        undo_btn = QPushButton("Undo  [z]")
        clear_btn = QPushButton("Clear  [w]")
        self.predict_btn = QPushButton("Predict  [p]")

        undo_btn.clicked.connect(self.canvas.undo_stroke)
        clear_btn.clicked.connect(self.canvas.clear)
        self.predict_btn.clicked.connect(self.predict)
        self.model_combo.currentTextChanged.connect(self.on_model_changed)

        topk_font = QFont(self.predict_btn.font())
        topk_font.setBold(True)
        topk_font.setFamily("Noto Sans Myanmar")
        self.topk_label.setFont(topk_font)

        controls = QHBoxLayout()
        controls.addWidget(undo_btn)
        controls.addWidget(clear_btn)
        controls.addWidget(self.predict_btn)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Model"))
        layout.addWidget(self.model_combo)
        layout.addWidget(self.model_status)
        layout.addWidget(self.status_label)
        layout.addWidget(QLabel("Prediction:"))
        layout.addWidget(self.prediction_label)
        layout.addWidget(QLabel(f"Top {TOP_K} candidates:"))
        layout.addWidget(self.topk_label)
        layout.addWidget(self.canvas, stretch=1)
        layout.addLayout(controls)
        self.setLayout(layout)

        for seq, slot in (
            ("Z", self.canvas.undo_stroke),
            ("W", self.canvas.clear),
            ("P", self.predict),
            ("Return", self.predict),
        ):
            sc = QShortcut(QKeySequence(seq), self)
            sc.setContext(Qt.WindowShortcut)
            sc.activated.connect(slot)

        QTimer.singleShot(0, self.load_current_model)

    def _show_loaded(self, name: str) -> None:
        p = self.predictors[name]
        ckpt_name = os.path.basename(p.local_checkpoint_path)
        self.model_status.setText(f"Loaded {name}: {ckpt_name}")
        self.model_status.setToolTip(
            f"{p.local_checkpoint_path}\n"
            f"vocab={p.codec.vocab_size} chars, device={self.device}"
        )
        self.status_label.setText("")
        self.predict_btn.setEnabled(True)

    def load_current_model(self) -> None:
        name = self.model_combo.currentText()
        if name in self.predictors:
            self._show_loaded(name)
            return
        if name in self._loading:
            self.model_status.setText(f"Loading {name}... (large models may take a minute)")
            self.predict_btn.setEnabled(False)
            return

        path = self.checkpoints[name]
        self._loading.add(name)
        self.model_status.setText(f"Loading {name}... (download may take a while)")
        self.status_label.setText("Please wait — window stays responsive")
        self.predict_btn.setEnabled(False)

        worker = ModelLoadWorker(
            name,
            self.labels_path,
            path,
            self.device,
            hf_token=self.hf_token,
        )
        worker.finished.connect(self._on_model_loaded)
        worker.failed.connect(self._on_model_failed)
        worker.finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        self._workers[name] = worker
        worker.start()

    def _on_model_loaded(self, name: str, predictor: StrokePredictor) -> None:
        self._loading.discard(name)
        self._workers.pop(name, None)
        self.predictors[name] = predictor
        if self.model_combo.currentText() == name:
            self._show_loaded(name)

    def _on_model_failed(self, name: str, message: str) -> None:
        self._loading.discard(name)
        self._workers.pop(name, None)
        if self.model_combo.currentText() == name:
            self.model_status.setText(f"Failed to load {name}")
            self.predict_btn.setEnabled(False)
            QMessageBox.critical(self, "Load error", message)

    def on_model_changed(self, _name: str) -> None:
        self.prediction_label.setText("—")
        self.topk_label.setText("")
        self.status_label.setText("")
        self.load_current_model()

    def _format_topk(self, topk: list[tuple[str, float]]) -> str:
        if not topk:
            return "—"
        parts = []
        for text, prob in topk:
            label = text if text else "∅"
            parts.append(f"{label} - {prob * 100:.0f}%")
        return ", ".join(parts)

    def predict(self) -> None:
        if not self.canvas.strokes:
            QMessageBox.warning(self, "Error", "Draw something first")
            return

        name = self.model_combo.currentText()
        if name not in self.predictors:
            QMessageBox.warning(self, "Error", f"{name} still loading — please wait")
            return
        try:
            topk = self.predictors[name].predict_topk(self.canvas.strokes)
            best = topk[0][0] if topk else ""
            self.prediction_label.setText(best or "—")
            self.topk_label.setText(self._format_topk(topk))
            self.status_label.setText(f"Predicted with {name}")
        except Exception as e:
            QMessageBox.critical(self, "Predict error", str(e))


def main() -> None:
    from dotenv import load_dotenv

    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Myanmar syllable handwriting predictor (PyQt5)"
    )
    parser.add_argument(
        "--file",
        required=True,
        help="label file (utf-8), same as training — e.g. syl.txt",
    )
    parser.add_argument(
        "--checkpoint_rnn",
        default=None,
        help="rnn checkpoint: local .pt path or hf://repo_id/filename.pt",
    )
    parser.add_argument(
        "--checkpoint_bilstm",
        default=None,
        help="bilstm checkpoint: local .pt path or hf://repo_id/filename.pt",
    )
    parser.add_argument(
        "--hf_token",
        default=None,
        help="Hugging Face token for private repos",
    )
    parser.add_argument(
        "--model",
        choices=["rnn", "bilstm"],
        default="bilstm",
        help="default model in dropdown",
    )
    parser.add_argument("--font_size", type=int, default=48)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cpu or cuda",
    )
    args = parser.parse_args()

    checkpoints: dict[str, str] = {}
    if args.checkpoint_rnn:
        checkpoints["rnn"] = args.checkpoint_rnn
    if args.checkpoint_bilstm:
        checkpoints["bilstm"] = args.checkpoint_bilstm
    if not checkpoints:
        parser.error("provide at least one of --checkpoint_rnn / --checkpoint_bilstm")

    default_model = args.model if args.model in checkpoints else next(iter(checkpoints))
    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    app = QApplication(sys.argv)
    win = PredictorWindow(
        labels_path=args.file,
        checkpoints=checkpoints,
        default_model=default_model,
        font_size=args.font_size,
        device=args.device,
        hf_token=hf_token,
    )
    win.setFixedSize(WINDOW_WIDTH, WINDOW_HEIGHT)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()