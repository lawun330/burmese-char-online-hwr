# Online Burmese Handwritten Character Recognition

## Overview
This project focuses on online Burmese handwritten character recognition using real human handwriting and compares three machine learning models: Recurrent Neural Network (RNN), Bidirectional Long Short-Term Memory (Bi-LSTM), and Liquid Neural Network (LNN).

## File Structure
```
/
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