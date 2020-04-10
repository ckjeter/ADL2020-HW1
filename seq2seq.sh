#!/usr/bin/env bash

python src/preprocess_seq2seq.py --output_dir datasets/seq2seq --data_dir $1 --nowtraining False
python src/eval_seq2seq.py --input datasets/seq2seq/data.pkl --output $2 --batch_size 64