import torch
from torch.utils.data import Dataset, DataLoader
import pickle
from utils import pad_to_len

class extractive_dataset(Dataset):
    def __init__(self, text, label, mask):
        self.text = text
        self.label = label
        self.mask = mask
    def __getitem__(self, index):
        text = self.text[index]
        label = self.label[index]
        mask = self.mask[index]
        text = torch.FloatTensor(text)
        label = torch.FloatTensor(label)
        mask = torch.BoolTensor(mask)
        return text, label, mask
    def __len__(self):
        return len(self.text)


class Seq2SeqDataset(Dataset):
    def __init__(self, data, padding=0,
                 max_text_len=300, max_summary_len=80):
        self.data = data
        self.padding = padding
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'id': self.data[index]['id'],
            'text': self.data[index]['text'][:self.max_text_len],
            'summary': self.data[index]['summary'][:self.max_summary_len],
            'len_text': len(self.data[index]['text']),
            'len_summary': len(self.data[index]['summary']),
            'attention_mask': [True] * min(len(self.data[index]['text']),
                                           self.max_text_len)
        }

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'len_text', 'len_summary']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'summary', 'attention_mask']:
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len, self.padding
            )
            batch[key] = torch.tensor(padded)

        return batch

class Seq2SeqDataset_test(Dataset):
    def __init__(self, data, padding=0,
                 max_text_len=300, max_summary_len=80):
        self.data = data
        self.padding = padding
        self.max_text_len = max_text_len
        self.max_summary_len = max_summary_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return {
            'id': self.data[index]['id'],
            'text': self.data[index]['text'][:self.max_text_len],
            #'summary': self.data[index]['summary'][:self.max_summary_len],
            'len_text': len(self.data[index]['text']),
            #'len_summary': len(self.data[index]['summary']),
            'attention_mask': [True] * min(len(self.data[index]['text']),
                                           self.max_text_len)
        }

    def collate_fn(self, samples):
        batch = {}
        for key in ['id', 'len_text']:
            batch[key] = [sample[key] for sample in samples]

        for key in ['text', 'attention_mask']:
            to_len = max([len(sample[key]) for sample in samples])
            padded = pad_to_len(
                [sample[key] for sample in samples], to_len, self.padding
            )
            batch[key] = torch.tensor(padded)

        return batch