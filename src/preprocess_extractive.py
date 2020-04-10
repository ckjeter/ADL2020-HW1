import torch
from tqdm import tqdm
import numpy as np
import nltk
import json
import pickle
import argparse
def loaddata_extractive(path, word_dim, embedd, num):
    zero_count = 0
    one_count = 0
    #length = 400
    text_all = []
    label_all = []
    mask_all = []
    with open(path, 'r') as r:
        for j, lines in enumerate(tqdm(r, total=num)):
            d = json.loads(lines)
            text = d['text']
            text_vec = []
            label = []
            mask = []
            for i, sent_bounds in enumerate(d['sent_bounds']):
                sent = nltk.word_tokenize(text[sent_bounds[0]: sent_bounds[1]])
                for word in sent:
                    try:
                        vec = embedd[word.lower()]
                        if len(vec) != int(word_dim):
                            text_vec.append(vec + [0] * (int(word_dim) - len(vec)))
                        else:
                            text_vec.append(vec)                        
                    except:
                        text_vec.append([0] * int(word_dim))
                        label.append([0]) #取搜不到的字 (可能是重要的專有名詞)
                        continue

                    if i == d['extractive_summary']:
                        label.append([1])
                    else:
                        label.append([0])
            zero_count += label.count([0])
            one_count += label.count([1])
            if len(text_vec) > 400:
                text_vec = text_vec[:400]
                label = label[:400]
                mask = [[1]] * 400
            else:
                mask = [[1]] * len(label) + [[0]] * (400 - len(label))
                text_vec += [[0]*int(word_dim)] * (400 - len(text_vec))
                label += [[0]] * (400 - len(label))
            #text_vec = torch.FloatTensor(text_vec)
            #label = torch.FloatTensor(label)
            text_all.append(text_vec)
            label_all.append(label)
            mask_all.append(mask)
    print("weight: {} / {} = {}".format(zero_count , one_count, zero_count/one_count))
    return text_all, label_all, mask_all
