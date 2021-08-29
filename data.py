from transformers import AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import torch
from pytorch_lightning import LightningDataModule
import json
import random

class DialogueDataset(Dataset):

    def __init__(self, data_dir: str, max_len = 256, data_ratio = 1):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained('vblagoje/bert-english-uncased-finetuned-pos',use_fast=False)
        with open(data_dir,'r') as f:
            data = f.read()
            data = json.loads(data)
        if data_dir.endswith('train_set.json'): # data_ratio is only usable in train et
            random.shuffle(data)
            self.data = data[:round(data_ratio*len(data))]
        else:
            self.data = data
        self.max_len = max_len

    
    def __getitem__(self, index: int):
        # in our setting we don't want one turn to be truncated into multiple. Thus 
        # we make sure that value of max_len is larger than max length of any input.

        # here inputs are tokens.
        sequence, label = self.data[index]
        sequence_padded = sequence + ['[PAD]']*(self.max_len-len(sequence))
        attention_mask = [1]*len(sequence)+[0]*(self.max_len-len(sequence))
        input_ids = self.tokenizer.convert_tokens_to_ids(sequence_padded)

        label_padded = label + ['O']*(self.max_len-len(label))
        mapper = {'O':0,'B':1,'I':2}
        labels = [mapper[item] for item in label_padded]

        first_sep = sequence_padded.index('[SEP]')
        first_pad = sequence_padded.index('[PAD]')
        token_type_ids = [0]*(first_sep+1) + [1]*(first_pad-first_sep-1) + [0]*(self.max_len-len(sequence))

        return {'input_ids':torch.tensor(input_ids,dtype=torch.long), 
                'attention_mask':torch.tensor(attention_mask,dtype=torch.long),
                'labels':torch.tensor(labels,dtype=torch.long),
                'token_type_ids':torch.tensor(token_type_ids,dtype=torch.long)}
    
    def __len__(self):
        return len(self.data)



class DialogueDataModule(LightningDataModule):

    def __init__(self, max_len, data_ratio):
        super().__init__()
        self.max_len = max_len
        self.data_ratio = data_ratio

    def setup(self, stage=None):
        # 指定在数据加载器中使用的训练/验证数据集
        if stage == 'fit' or stage is None:
            self.train_set = DialogueDataset('./dataset/train_set.json',max_len = self.max_len, data_ratio = self.data_ratio)
            self.dev_set = DialogueDataset('./dataset/dev_set.json',max_len = self.max_len, data_ratio = 1)

        # 指定测试数据集用于dataloader
        if stage == 'test' or stage is None:
            self.test_set = DialogueDataset('./dataset/test_set.json',max_len = self.max_len, data_ratio = 1)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=128, num_workers=32)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=128, num_workers=32)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=8,num_workers=32)