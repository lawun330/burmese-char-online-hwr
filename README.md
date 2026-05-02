# Online Burmese Handwritten Character Recognition

## Overview
This project focuses on online Burmese handwritten character recognition using real human handwriting and compares three machine learning models: Recurrent Neural Network (RNN) and Bidirectional Long Short-Term Memory (Bi-LSTM).

## File Structure
```
/
...
├── dataset/la_wun_nannda/
├── model/
│   ├── runs/
│   │   ├── bilstm/
│   │   └── rnn/
│   └── scripts/
│
├── convert2image.py     # originally Sayar's
├── dataset_browser.py   # originally Sayar's # modified to add check, rename, delete, search, refresh buttons
├── hw_collector.py      # originally Sayar's # modified to add undo button & shortcuts
│
├── static/              # added by Ko Sai Jack
├── Documentation.pdf    # added by Ko Sai Jack
├── Makefile             # added by Ko Sai Jack
├── index.html           # added by Ko Sai Jack
│
├── conda_environment.yaml
├── requirements-render.txt
├── requirements-s3.txt
├── requirements.txt     # added by Ko Sai Jack
│
├── server.py            # added by Ko Sai Jack
├── server_s3.py
│
├── auto_save_strokes_s3.py
├── sync_strokes_s3.py
│
└── syl.txt              # originally Sayar's
```

---

## Dataset
This project uses custom dataset of real human handwriting, defined by symbols in `syl.txt`, with each symbol written four times.

## Data Collection
This project implements a multi-mode handwritten Burmese character data collector. Stroke data can be captured in localhost mode on PC or mobile for direct local storage, and the production path is backed by an S3 + Render cloud pipeline for centralized dataset persistence. Local and cloud datasets are connected through built-in synchronization utilities, so collected strokes remain consistent and ready for shared access and model training.

## Handwriting Data Collector
- Render: https://burmese-char-online-hwr.onrender.com

---

## Models

### 1. RNN + CTC

#### Architecture:
- Input stroke
- Hidden neurons (linear projection)
- ReLU activation function
- RNN neurons x num_layers
- Dropout regularization (only if num_layers > 1)
- Output neurons/logits with log_softmax
- Loss function of the output: CTC

#### Commands:
```python
# BASELINE
python main.py --model rnn --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 1 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.001

# EXPERIMENT 1: increased to 2 layers
python main.py --model rnn --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.001

# EXPERIMENT 2: increased to 2 layers, double hidden units
python main.py --model rnn --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 256 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.001

# EXPERIMENT 3: increased to 2 layers, double batch size
python main.py --model rnn --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 64 --lr 0.001

# EXPERIMENT 4: increased to 2 layers, double batch size, double epochs
python main.py --model rnn --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 60 --batch_size 64 --lr 0.001

# EXPERIMENT 5: increased to 2 layers, half learning rate
python main.py --model rnn --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.0005
```

#### Best metric achieved during training:

| Index | best val loss (@ep) | best val CER (@ep) | best val WER (@ep) | train/val loss | val error rate |
|------:|--------------------:|-------------------:|-------------------:|---------------:|---------------:|
| Baseline     |  2.521 (29) |  0.745 (30) |  0.988 (30) | ![](model/runs/rnn/20260502-185639/loss.png) | ![](model/runs/rnn/20260502-185639/error_rates.png) |
| Experiment 1 |  2.508 (29) |  0.737 (30) |  0.985 (28) | ![](model/runs/rnn/20260502-190606/loss.png) | ![](model/runs/rnn/20260502-190606/error_rates.png) |
| Experiment 2 |  2.666 (28) |  0.803 (24) |  0.989 (19) | ![](model/runs/rnn/20260502-191453/loss.png) | ![](model/runs/rnn/20260502-191453/error_rates.png) |
| Experiment 3 |  2.583 (30) |  0.764 (30) |  0.986 (28) | ![](model/runs/rnn/20260502-195246/loss.png) | ![](model/runs/rnn/20260502-195246/error_rates.png) |
| Experiment 4 |  **2.459** (60) |  **0.729** (55) |  **0.982** (58) | ![](model/runs/rnn/20260502-201957/loss.png) | ![](model/runs/rnn/20260502-201957/error_rates.png) |
| Experiment 5 |  2.741 (30) |  0.824 (26) |  0.993 (27) | ![](model/runs/rnn/20260502-204005/loss.png) | ![](model/runs/rnn/20260502-204005/error_rates.png) |

- best val loss = minimum va_loss over all epochs.
- best val CER = minimum va_cer over all epochs.
- best val WER = minimum va_wer over all epochs.

### 2. LSTM + CTC

#### Architecture:
- Input stroke
- Hidden neurons (linear projection)
- ReLU activation function
- LSTM neurons x num_layers
- Bidirection
- Dropout regularization (only if num_layers > 1)
- Output neurons/logits with log_softmax
- Loss function of the output: CTC

#### Commands:
```python
# BASELINE
python main.py --model bilstm --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 1 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.001

# EXPERIMENT 1: increased to 2 layers
python main.py --model bilstm --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.001

# EXPERIMENT 2: increased to 2 layers, double hidden units
python main.py --model bilstm --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 256 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.001

# EXPERIMENT 3: increased to 2 layers, double batch size
python main.py --model bilstm --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 64 --lr 0.001

# EXPERIMENT 4: increased to 2 layers, double batch size, double epochs
python main.py --model bilstm --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 60 --batch_size 64 --lr 0.001

# EXPERIMENT 5: increased to 2 layers, half learning rate
python main.py --model bilstm --mode train --plot yes --data dataset/ --split_ratio 0.2 --labels syl.txt --seed 42 --hidden 128 --num_layers 2 --dropout 0.01 --epochs 30 --batch_size 32 --lr 0.0005
```

#### Best metric achieved during training:

| Index | best val loss (@ep) | best val CER (@ep) | best val WER (@ep) | train/val loss | val error rate |
|------:|--------------------:|-------------------:|-------------------:|---------------:|---------------:|
| Baseline     |  1.140 (30) |  0.358 (30) |  0.761 (30) | ![](model/runs/bilstm/20260502-225441/loss.png) | ![](model/runs/bilstm/20260502-225441/error_rates.png) |
| Experiment 1 |  0.763 (30) |  0.231 (30) |  0.580 (30) | ![](model/runs/bilstm/20260502-234814/loss.png) | ![](model/runs/bilstm/20260502-234814/error_rates.png) |
| Experiment 2 |  **0.301** (30) |  **0.093** (30) |  **0.252** (29) | ![](model/runs/bilstm/20260503-000239/loss.png) | ![](model/runs/bilstm/20260503-000239/error_rates.png) |
| Experiment 3 |  1.050 (30) |  0.326 (30) |  0.738 (30) | ![](model/runs/bilstm/20260503-003533/loss.png) | ![](model/runs/bilstm/20260503-003533/error_rates.png) |
| Experiment 4 |  0.407 (60) |  0.125 (59) |  0.344 (58) | ![](model/runs/bilstm/20260503-005006/loss.png) | ![](model/runs/bilstm/20260503-005006/error_rates.png) |
| Experiment 5 |  1.114 (30) |  0.352 (30) |  0.755 (29) | ![](model/runs/bilstm/20260503-011823/loss.png) | ![](model/runs/bilstm/20260503-011823/error_rates.png) |

- best val loss = minimum va_loss over all epochs.
- best val CER = minimum va_cer over all epochs.
- best val WER = minimum va_wer over all epochs.

---

## References

- CTC Loss:
    - Ogun, S. O. (2020, July 17). _Breaking down the CTC loss_. Sewade's Website. https://ogunlao.github.io/blog/2020/07/17/breaking-down-ctc-loss.html
    - DataMListic. (2022, Aug 19). _Connectionist temporal classification (CTC) explained_ [Video]. YouTube. https://youtu.be/jDPl1QJGLpE