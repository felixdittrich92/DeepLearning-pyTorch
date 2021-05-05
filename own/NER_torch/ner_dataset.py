import os

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizerFast

os.environ["TOKENIZERS_PARALLELISM"] = "true"
FILE_PATH = 'data/own.txt'

def read_conll_from_txt_to_df(file_path):  
    df = pd.DataFrame(columns=['SENTENCE', 'TOKEN', 'LABEL'])
    sent = 1

    with open(file_path, 'r') as f:
        for i, line in enumerate(f):
            line = line.replace('\n', '')
            splitted = line.split()
            if not splitted:
                sent += 1
            else:
                df.loc[i] = [sent, splitted[0], splitted[1]]   
    return df       

data = read_conll_from_txt_to_df(FILE_PATH)

class SentenceGetter(object):

    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(token, tag) for token, tag in zip(s["TOKEN"].values.tolist(),
                                                                 s["LABEL"].values.tolist())]
        self.grouped = self.data.groupby("SENTENCE").apply(agg_func)
        self.sentences = [s for s in self.grouped]

    def get_next(self):
        try:
            s = self.grouped["SENTENCE: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(data)

tags_vals = list(set(data["LABEL"].values))
tag2index = {t: i for i, t in enumerate(tags_vals)}
index2tag = {i: t for i, t in enumerate(tags_vals)}  
sentences = [' '.join([s[0] for s in sent]) for sent in getter.sentences]

labels = [[s[1] for s in sent] for sent in getter.sentences]
#labels = [[tag2idx.get(l) for l in lab] for lab in labels]



##### only overview #####
tags = ["[PAD]"]
tags.extend(list(set(data["LABEL"].values)))
tag2idx = {t: i for i, t in enumerate(tags)}
print('Length of Labels : ' + str(len(tags)))

words = ["[PAD]", "[UNK]"]
words.extend(list(set(data["TOKEN"].values)))
word2idx = {t: i for i, t in enumerate(words)}
print('Length of unique words : ' + str(len(words)))

# check dataset
def dataset_checker(sent_list, label_list):
    sent_check = list()
    for el in sent_list:
        sent_check.append(len(el.split()))
    label_check = list()
    for el in label_list:
        label_check.append(len(el))

    for index, (first, second) in enumerate(zip(sent_check, label_check)):
        if first != second:
            print(index, second)
dataset_checker(sentences, labels)
##### only overview #####

unique_tags = list(set(tag for text in labels for tag in text))

train_sent, test_sent, train_label, test_label = train_test_split(sentences, labels, test_size=0.05)
train_sent, val_sent, train_label, val_label = train_test_split(train_sent, train_label, test_size=0.15)
    
print('FULL DATASET SENT: ' + str(len(sentences)))
print('FULL DATASET LABELS: ' + str(len(labels)))
print('Train sent size : ' + str(len(train_sent)))
print('Train label size : ' + str(len(train_label)))
print('Test sent size : ' + str(len(test_sent)))
print('Test label size : ' + str(len(test_label)))
print('Val sent size : ' + str(len(val_sent)))
print('Val label size : ' + str(len(val_label)))

print('Check Training dataset')
dataset_checker(train_sent, train_label)
print('Check Test dataset')
dataset_checker(test_sent, test_label)
print('Check Validation dataset')
dataset_checker(val_sent, val_label)

class ConllDataset(Dataset):
    def __init__(self, tokenizer, sentences, labels, max_len):
        self.len = len(sentences)
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sentence = self.sentences[index].strip().split()
        labels = self.labels[index]

        inputs = self.tokenizer.encode_plus(
            sentence,
            None,
            is_split_into_words=True,
            add_special_tokens=True,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_offsets_mapping=True,
            return_token_type_ids=False,
            return_attention_mask=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']


        labels = [tag2index[label] for label in labels] 
        
        # code based on https://huggingface.co/transformers/custom_datasets.html#tok-ner
        # create an empty array of -100 of length max_length
        encoded_labels = np.ones(len(inputs["offset_mapping"]), dtype=int) * -100
        
        # set only labels whose first offset position is 0 and the second is not 0
        i = 0
        for idx, mapping in enumerate(inputs["offset_mapping"]):
          if mapping[0] == 0 and mapping[1] != 0:
            # overwrite label
            encoded_labels[idx] = labels[i]
            i += 1
        
        inputs.pop('offset_mapping')
        
        return {
            'ids': torch.tensor(ids).flatten(),
            'mask': torch.tensor(mask).flatten(),
            'tags': torch.tensor(encoded_labels)
        } 

class NERConllDataset(pl.LightningDataModule):
    
    def __init__(self, tokenizer, train_sent, train_label, val_sent, val_label, test_sent, test_label, max_len, batch_size):
        super().__init__()
        self.tokenizer = tokenizer
        self.train_sent = train_sent
        self.train_label = train_label
        self.val_sent = val_sent
        self.val_label = val_label
        self.test_sent = test_sent
        self.test_label = test_label
        self.max_len = max_len
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.train_dataset = ConllDataset(self.tokenizer, self.train_sent, self.train_label, self.max_len)
        
        self.val_dataset = ConllDataset(self.tokenizer, self.val_sent, self.val_label, self.max_len)
        
        self.test_dataset = ConllDataset(self.tokenizer, self.test_sent, self.test_label, self.max_len)
    
    def eval_print(self, pos: int):
        for token, label in zip(self.tokenizer.convert_ids_to_tokens(self.train_dataset[pos]["ids"].numpy()), self.train_dataset[pos]["tags"].numpy()):
            print('{0:10}  {1}'.format(token, label))

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
        
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=8)
    

MAX_LEN = 256
BATCH_SIZE = 8
tokenizer = BertTokenizerFast.from_pretrained('bert-base-german-cased', do_lower_case=False)
# save tokenizer
tokenizer.save_pretrained("model/tokenizer/")

data_module = NERConllDataset(tokenizer, train_sent, train_label, val_sent, val_label, test_sent, test_label, max_len=MAX_LEN, batch_size=BATCH_SIZE)
data_module.setup()
data_module.eval_print(pos=5)
train = data_module.train_dataloader()
val = data_module.val_dataloader()
test = data_module.test_dataloader()


