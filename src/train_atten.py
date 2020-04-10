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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure(figsize=(attentions.shape))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.cpu().detach().numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words.split(' '))

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.savefig("attention.png")

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
'''
class Decoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        with open('datasets/seq2seq/embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.output_dim = 113377
        self.query_adjust = nn.Linear(600, 300)
        #self.attn = nn.Linear(600, 1)
        self.rnn = nn.GRU(300, 150, 2, dropout = dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(600, self.output_dim)
        self.softmax = nn.LogSoftmax(dim=2)
        #self.fc = nn.Linear(100, 1)
        
    def forward(self, idxs, state, encoder_output): 
        #state: query -> batch_size * 600
        #encoder_output: key, value -> batch_size * text_length * 300
        embed = self.embedding(idxs) #batch_size * 1 * 300
        query = state.transpose(0, 1).reshape(-1, 600)
        query = self.query_adjust(query) # batch_size * 300
        #query = query.unsqueeze(1).repeat(1, encoder_output.shape[1], 1) #batch_size * textlength * 300
        #atten = F.softmax(self.attn(torch.cat((query, encoder_output), 2)), dim=1)
        atten = F.softmax(torch.bmm(encoder_output, query.unsqueeze(2)), dim=1)
        context = torch.bmm(atten.transpose(1, 2), encoder_output)
        output, state = self.rnn(embed, state)
        output = torch.cat((output, context), 2)
        prediction = self.softmax(self.fc(output))
        return prediction, state
'''
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        with open('datasets/seq2seq/embedding.pkl', 'rb') as f:
            embedding = pickle.load(f)
        embedding_weight = embedding.vectors
        self.embedding = nn.Embedding.from_pretrained(embedding_weight)
        self.rnn = nn.GRU(300, 300, 1, batch_first=True)
        
    def forward(self, idxs, state): 
        embed = self.embedding(idxs)
        output, state = self.rnn(embed, state)
        return output, state

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.output_dim = 113377
        self.softmax = nn.Softmax(dim=1)
        self.fc1 = nn.Linear(600, 300)
        self.tanh = nn.Tanh()
        self.fc2 = nn.Linear(600, self.output_dim)
        #self.fc3 = nn.Linear(300, 300)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, data, nowtraining, draw):
        text = data['text'].to(self.device)
        o, state = self.encoder(text)
        state = state.transpose(0, 1)
        state = state.reshape(-1, 600)
        state = self.fc1(state)
        state = self.tanh(state).unsqueeze(0)
        #first input to the decoder is the <sos> tokens
        input = torch.ones((text.shape[0], 1), dtype=torch.long).to(self.device) #<s>
        if nowtraining:      
            target = data['summary'].to(self.device)
            query = torch.zeros((target.shape[0], target.shape[1], 300)).to(self.device)
            for t in range(0, target.shape[1]):
                output, state = self.decoder(input, state)
                output = output.transpose(0, 1)
                query = query.transpose(0, 1)
                query[t] = output
                output = output.transpose(0, 1)
                query = query.transpose(0, 1)         
                input = target.transpose(0, 1)[t].unsqueeze(1)
            atten = F.softmax(torch.bmm(query, o.transpose(1, 2)), dim=2)
            atten = torch.bmm(atten, o) #32, 80, 150
            predict = self.fc2(torch.cat((atten, query), 2))
            predict = self.softmax(predict)
            return predict
        else:
            with open('datasets/seq2seq/embedding.pkl', 'rb') as f:
                embedding = pickle.load(f)
            predict = []
            for batch in range(0, text.shape[0]):
                #atten_all = []
                summary = ''
                _o = o[batch].unsqueeze(0)
                output, _state = self.decoder(input[batch].unsqueeze(0), state.transpose(0, 1)[batch].unsqueeze(1).contiguous())
                atten_weight = F.softmax(torch.bmm(output, _o.transpose(1, 2)), dim=2)
                #atten_all.append(atten_weight.view(-1))
                atten = torch.bmm(atten_weight, _o)
                output = self.fc2(torch.cat((atten, output), 2))
                output = self.softmax(output)
                for t in range(1, 80):
                    #output.shape : batch_size, 1, 113377
                    top1 = output.argmax(2)
                    summary += embedding.vocab[top1.view(-1)] + ' '
                    #summary.append(embedding.vocab[top1.view(-1)])
                    if top1.view(-1) == 2: #</s>
                        break
                    _input = top1
                    output, _state = self.decoder(_input, _state)
                    atten_weight = F.softmax(torch.bmm(output, _o.transpose(1, 2)), dim=2)
                    #atten_all.append(atten_weight.view(-1))
                    atten = torch.bmm(atten_weight, _o)
                    output = self.fc2(torch.cat((atten, output), 2))
                    output = self.softmax(output)
                '''
                if batch == 1 and draw:
                    truetext = [embedding.vocab[i] for i in text[batch]]
                    atten_all = torch.stack(atten_all)
                    showAttention(truetext, summary, atten_all)
                    draw = False
                '''
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
    decoder = Decoder()
    model = Seq2Seq(encoder, decoder, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr = lr)
    #criterion = nn.CrossEntropyLoss(ignore_index=0).to(device)
    criterion = nn.NLLLoss(ignore_index=0).to(device)

    for epoch in range(epochs):
        train_loss = 0.0
        valid_loss = 0.0
        print("Epoch: {}/{}".format(epoch + 1, epochs))
        for batch, data in enumerate(trainloader):
            model.zero_grad()
            output = model(data, True, False).view(-1, 113377)
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
        savepath = 'model/atten'+str(epoch)+'.pth'
        torch.save(model.state_dict(), savepath)     
        logging.info("Save Model: {}".format(savepath))
        for batch, data in enumerate(validloader):
            model.zero_grad()
            output = model(data, True, False).view(-1, 113377)
            target = data['summary'].cuda()
            target = target.view(-1)
            loss = criterion(output, target)
            # print statistics
            valid_loss += loss.item()
            sys.stdout.write("    Valid Batch: {}/{}, Batch Loss: {}\r".format(batch, len(validloader), loss.item()))
            sys.stdout.flush()
        logging.info("Validation Loss: {}".format(valid_loss  / len(validloader)))
        if valid_loss > train_loss:
            for param_group in optimizer.param_groups:
                param_group['weight_decay'] = 0.05
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001


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