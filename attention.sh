#!/usr/bin/env bash

python src/preprocess_seq2seq.py --nowtraining False --output_dir datasets/seq2seq --data_dir $1
python src/eval_atten.py --input datasets/seq2seq/data.pkl --output $2 --batch_size 64