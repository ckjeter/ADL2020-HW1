import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from tqdm import tqdm
import numpy as np
import json
import nltk
import dataset
import sys
from train_extractive import Model
import argparse
import matplotlib.pyplot as plt
'''
def preprocess(test, embedd):
    id = []
    text_all = []
    sent_all = []
    with open(test, 'r') as r:
        for i, line in enumerate(tqdm(r, total=20000)):
            text = []
            sentid = []
            d = json.loads(line)
            id.append(d['id'])
            _text = d['text']
            for j, sent_bounds in enumerate(d['sent_bounds']):
                sent = nltk.word_tokenize(_text[sent_bounds[0]: sent_bounds[1]])
                for word in sent:
                    try:
                        text.append(embedd[word.lower()])
                    except:
                        text.append([0]*300)
                sentid += [len(sent)]
            if len(text) <= 1:
                continue
            if len(text) > 400:
                text = text[:400]
                #sentid = sentid[:400]
            else:
                text += [[0]*300] * (400 - len(text))
                sentid += [-1] * (400 - len(sentid))
            text_all.append(text)
            sent_all.append(sentid)
    return id, text_all, sent_all
def calculate(id, text, sent, result_path, model):
    with open(result_path, 'w') as w:
        for i in tqdm(range(0, len(id))):
            result = {}
            output = model(torch.FloatTensor([text[i]]).cuda()).view(-1)
            start = 0
            predict = -1
            baseline = 0
            for j, count in enumerate(sent[i]):
                if start >= 400:
                    break
                score = 0
                for k in range(count):
                    score += output[start + k]
                if score/count > baseline:
                    baseline = score/count
                    predict = j
                start += count
            result['id'] = id[i]
            result['predict_sentence_index'] = [predict]
            _result = json.dumps(result)
            w.write(_result + '\n')
'''
def calculate(test, result, word_dim, model, embedd):
    loc = []
    with open(test, 'r') as r, open(result, 'w') as w:
        for i, line in enumerate(tqdm(r, total=20000)):
            result = {}
            d = json.loads(line)
            result['id'] = d['id']
            _text = d['text']

            sentence_token = []
            for j, sent_bounds in enumerate(d['sent_bounds']):
                sentence_token.append(nltk.word_tokenize(_text[sent_bounds[0]: sent_bounds[1]]))
            if len(sentence_token) <= 1:
                result['predict_sentence_index'] = [0]
                result = json.dumps(result)
                w.write(result + '\n')  
                continue             

            text_vec = []
            for sent in sentence_token:
                for word in sent:
                    try:
                        text_vec.append(embedd[word.lower()])
                    except:
                        text_vec.append([0]*int(word_dim))
            
            output = model(torch.FloatTensor([text_vec]).cuda()).view(-1, 1)
            #output = torch.sigmoid(output).view(-1, 1)

            start = 0
            predict = -1
            baseline = 0
            for j, sent in enumerate(sentence_token):
                score = 0
                for k, word in enumerate(sent):
                    score += output[start + k]
                if score/len(sent) > baseline:
                    baseline = score/len(sent)
                    predict = j
                start += len(sent)
            result['predict_sentence_index'] = [predict]
            if predict >= 0:
                loc.append(predict / len(d['sent_bounds']))
            result = json.dumps(result)
            w.write(result + '\n')
    return loc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--word_dim', type=int, default = 300)
    parser.add_argument('--relative_location', type=bool, default = False)
    parser.add_argument('--input', type=str, default = False)
    parser.add_argument('--output', type=str, default = False)

    args = parser.parse_args()

    word_dim = str(args.word_dim)
    test_path = args.input
    result_path = args.output
    draw = args.relative_location
    print("input: {}".format(test_path))
    print("output: {}".format(result_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(int(word_dim), 0).to(device)
    model.load_state_dict(torch.load('model/Extractive.pth'))
    model.eval()
    embedd = {}
    print("Loading embedding...")
    with open('glove/glove.6B.'+word_dim+'d.txt', 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedd[word] = coefs
    #print("Preparing data...")
    #id, text, sent = preprocess(test_path, embedd)
    print("Evaluating...")
    loc = calculate(test_path, result_path, word_dim, model, embedd)
    if draw:
        plt.hist(loc, bins=[i for i in np.arange(0, 1.02, 0.02)])
        plt.xlabel("Relative Location")
        plt.ylabel("Density")
        plt.savefig("relative_location.png")
    #calculate(id, text, sent, result_path, model)
if __name__ == "__main__":
    main()

