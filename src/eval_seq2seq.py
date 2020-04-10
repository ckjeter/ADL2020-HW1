import numpy as np
import logging
import pickle
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn.utils.rnn as rnn_utils
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import json
import nltk
import os
import sys
import argparse
from tqdm import tqdm 
from datetime import datetime
import dataset
from train_seq2seq import Encoder, Decoder, Seq2Seq

def main(args):
    inputpath = args.input
    predictpath = args.output
    batch_size = args.batch_size
    print("input: {}".format(inputpath))
    print("output: {}".format(predictpath))
    with open(inputpath, 'rb') as f:
        dataset = pickle.load(f) 
    dataloader = DataLoader(dataset, batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    logging.info("device : {}".format(device))
    encoder = Encoder(0)
    decoder = Decoder(0)
    model = Seq2Seq(encoder, decoder, device).to(device)
    model.load_state_dict(torch.load('model/s2s.pth'))
    model.eval()

    with open(predictpath, 'w') as w:
        for i, data in enumerate(tqdm(dataloader, total=len(dataloader))):
            summary = model(data, False)
            for j in range(0, len(data['id'])):
                predict = {}
                predict['id'] = data['id'][j]
                #summary = " ".join(model(data, False))
                predict['predict'] = summary[j]
                result = json.dumps(predict)
                w.write(result + '\n')

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = '64')
    parser.add_argument('--input', type=str, default = 'datasets/seq2seq/valid.pkl')
    parser.add_argument('--output', type = str, default = 'predict/predict_seq2seq.jsonl')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)