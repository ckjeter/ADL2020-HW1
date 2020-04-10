import numpy as np
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
import dataset
import preprocess_extractive

class Model(nn.Module):
    def __init__(self, word_dim, dropout):
        super(Model,self).__init__()
        #self.rnn = nn.RNN(word_dim, 100, 1, batch_first=True)
        self.gru = nn.GRU(word_dim, 100, 2, batch_first=True, bidirectional=True)
        #self.dropout = nn.Dropout(dropout)
        #self.lstm = nn.LSTM(200, 200, 2, batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(200, 1)
    
    def forward(self,input):
        #output, _ = self.rnn(input)
        output, _ = self.gru(input)
        #output = self.dropout(output)
        #output, _ = self.lstm(output)
        output = self.dropout(output)
        output = self.fc1(output)     
        return output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 16)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--dropout', type=float, default = 0.25)
    parser.add_argument('--word_dim', type=int, default = 300)

    args = parser.parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dropout = args.dropout
    word_dim = str(args.word_dim)

    embedd = {}
    with open('glove/glove.6B.'+word_dim+'d.txt', 'r', encoding='utf8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedd[word] = coefs
        
    print("Preprocessing: ")

    text, label, mask = preprocess_extractive.loaddata_extractive('datasets/train.jsonl', word_dim, embedd, 71604)
    train_data  = dataset.extractive_dataset(text, label, mask)
    trainloader = DataLoader(train_data, batch_size, shuffle=True)

    text, label, mask = preprocess_extractive.loaddata_extractive('datasets/valid.jsonl', word_dim, embedd, 20000)
    valid_data  = dataset.extractive_dataset(text, label, mask)
    validloader = DataLoader(valid_data, batch_size, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    print("device = {}".format(device))


    model = Model(int(word_dim), dropout)
    model.to(device)
    optimizer = optim.Adam(model.parameters(),lr = lr)
    criterion = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([7]).cuda())
    #criterion = nn.BCEWithLogitsLoss(reduction='mean')

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        for batch, data in enumerate(trainloader):
            text, label, mask = data
            #text = Variable(text).cuda()
            text = text.cuda()
            label = label.cuda()
            mask = mask.cuda()
            model.zero_grad()
            output = model(text)
            output = torch.masked_select(output, mask)
            label = torch.masked_select(label, mask)
            loss = criterion(output, label)
            loss.backward() 
            optimizer.step()
            # print statistics
            train_loss += loss.item()
            sys.stdout.write("    Train Batch: {}/{}, Batch Loss: {}\r".format(batch, len(trainloader), loss.item()))
            sys.stdout.flush()
        print("\n    Training Loss: {}".format(train_loss  / len(trainloader)))
        torch.save(model.state_dict(), 'model/Extractive'+str(epoch)+'.pth')     
        print("Save Model: model/Extractive{}.pth".format(str(epoch)))
        for batch, data in enumerate(validloader):
            text, label, mask = data
            text = text.cuda()
            label = label.cuda()
            mask = mask.cuda()
            model.zero_grad()
            output = model(text)
            output = torch.masked_select(output, mask)
            label = torch.masked_select(label, mask)
            loss = criterion(output, label)
            # print statistics
            valid_loss += loss.item()
            sys.stdout.write("    Valid Batch: {}/{}, Batch Loss: {}\r".format(batch, len(validloader), loss.item()))
            sys.stdout.flush()
        print("\n    Validation Loss: {}\n".format(valid_loss / len(validloader)))
        

if __name__ == "__main__":
    main()
