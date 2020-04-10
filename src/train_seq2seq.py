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
import dataset

class Encoder(nn.Module):
    def __init__(self, dropout) -> None:
        super(Encoder, self).__init__()
        with open('datasets/seq2seq/embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.GRU(300, 150, 2, dropout = dropout, batch_first=True, bidirectional=True)

    def forward(self, idxs):
        embed = self.embedding(idxs)
        output, state = self.rnn(embed)
        return output, state

class Decoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        with open('datasets/seq2seq/embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.output_dim = 113377
        self.rnn = nn.GRU(300, 150, 2, dropout = dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(300, self.output_dim)
        #self.fc = nn.Linear(100, 1)
        
    def forward(self, idxs, state): 
        embed = self.embedding(idxs)
        output, state = self.rnn(embed, state)
        prediction = self.fc(output)
        return prediction, state

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(150, 150)
        self.tanh = nn.Tanh()
        
    def forward(self, data, nowtraining):
        text = data['text'].to(self.device)
        o, state = self.encoder(text)
        state = self.fc1(state)
        state = self.tanh(state)
        #first input to the decoder is the <sos> tokens
        input = torch.ones((text.shape[0], 1), dtype=torch.long).to(self.device) #<s>
        if nowtraining:      
            target = data['summary'].to(self.device)
            outputs = torch.zeros((target.shape[0], target.shape[1], self.decoder.output_dim)).to(self.device)
            for t in range(0, target.shape[1]):
                output, state = self.decoder(input, state)
                #print(output.shape) #batch_size, 1, 113377
                output = output.transpose(0, 1)
                outputs = outputs.transpose(0, 1)
                #outputs[t] = self.softmax(output)
                outputs[t] = output
                output = output.transpose(0, 1)
                outputs = outputs.transpose(0, 1)         
                input = target.transpose(0, 1)[t].unsqueeze(1)
            return outputs
        else:
            with open('datasets/seq2seq/embedding.pkl', 'rb') as f:
                embedding = pickle.load(f)

            predict = []
            for batch in range(0, text.shape[0]):
                summary = ''
                output, _state = self.decoder(input[batch].unsqueeze(0), state.transpose(0, 1)[batch].unsqueeze(1).contiguous())
                for t in range(1, 80):
                    #output.shape : batch_size, 1, 113377
                    top1 = output.argmax(2)
                    summary += embedding.vocab[top1.view(-1)] + ' '
                    #summary.append(embedding.vocab[top1.view(-1)])
                    if top1.view(-1) == 2: #</s>
                        break
                    _input = top1
                    output, _state = self.decoder(_input, _state)
                predict.append(summary[4:-12])
            return predict




def main(args):
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    dropout = args.dropout
    word_dim = args.word_dim
    with open("datasets/seq2seq/train.pkl", 'rb') as f:
        trainset = pickle.load(f)
    with open("datasets/seq2seq/valid.pkl", 'rb') as f:
        validset = pickle.load(f) 
    trainloader = DataLoader(trainset, batch_size, shuffle=True, collate_fn=trainset.collate_fn) 
    validloader = DataLoader(validset, batch_size, shuffle=True, collate_fn=validset.collate_fn)   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = 'cpu'
    logging.info("device : {}".format(device))
    encoder = Encoder(dropout)
    decoder = Decoder(dropout)
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(),lr = lr)
    criterion = nn.CrossEntropyLoss().to(device)
    #criterion = nn.NLLLoss().to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        for batch, data in enumerate(trainloader):
            model.zero_grad()
            output = model(data, True).view(-1, 113377)
            target = data['summary'].cuda()
            target = target.view(-1)
            loss = criterion(output, target)
            loss.backward() 
            optimizer.step()
            # print statistics
            train_loss += loss.item()
            sys.stdout.write("    Train Batch: {}/{}, Batch Loss: {}\r".format(batch, len(trainloader), loss.item()))
            sys.stdout.flush()
        logging.info("Training Loss: {}".format(train_loss  / len(trainloader)))
        savepath = 'model/s2s'+str(epoch)+'.pth'
        torch.save(model.state_dict(), savepath)     
        logging.info("Save Model: {}".format(savepath))
        for batch, data in enumerate(validloader):
            model.zero_grad()
            output = model(data, True).view(-1, 113377)
            target = data['summary'].cuda()
            target = target.view(-1)
            loss = criterion(output, target)
            # print statistics
            valid_loss += loss.item()
            sys.stdout.write("    Valid Batch: {}/{}, Batch Loss: {}\r".format(batch, len(validloader), loss.item()))
            sys.stdout.flush()
        logging.info("Validation Loss: {}".format(valid_loss  / len(validloader)))


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 5)
    parser.add_argument('--epochs', type = int, default = 10)
    parser.add_argument('--lr', type = float, default = 0.001)
    parser.add_argument('--dropout', type=float, default = 0)
    parser.add_argument('--word_dim', type=int, default = 300)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = _parse_args()
    loglevel = os.environ.get('LOGLEVEL', 'INFO').upper()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=loglevel, datefmt='%Y-%m-%d %H:%M:%S')
    main(args)