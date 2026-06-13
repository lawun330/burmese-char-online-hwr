#!/bin/bash

# see models: https://huggingface.co/lawun330/burmese-char-online-hwr/

# prediction with model in local storage
# python hw_predictor.py \
#   --file syl.txt \
#   --checkpoint_rnn model/checkpoints/rnn_seed42_20260502-201957.pt \
#   --checkpoint_bilstm model/checkpoints/bilstm_seed42_20260503-000239.pt \
#   --model bilstm

# prediction with model in Hugging Face
python hw_predictor.py \
  --file syl.txt \
  --checkpoint_rnn hf://lawun330/burmese-char-online-hwr/rnn_seed42_20260502-201957.pt \
  --checkpoint_bilstm hf://lawun330/burmese-char-online-hwr/bilstm_seed42_20260502-225441.pt \
  --model rnn