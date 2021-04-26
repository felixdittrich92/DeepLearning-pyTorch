import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap

from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast
from tqdm.auto import tqdm

df = pd.read_csv("data/news_summary.csv", encoding="latin-1")
df = df[['text', 'ctext']]
df.colums = ['summary', 'text']
df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.15)

class NewsSummaryDataset(Dataset):
    
    def __init__(self, data: pd.Dataframe, tokenizer: T5TokenizerFast, text_max_token_len = 512, summary_max_token_len = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row['text']
        
        text_encoding = self.tokenizer(text, max_length = self.text_max_token_len, padding = 'max_length', truncation = True, return_attention_mask = True, add_special_tokens = True, return_tensors = 'pt')
        summary_encoding = self.tokenizer(text, max_length = self.summary_max_token_len, padding = 'max_length', truncation = True, return_attention_mask = True, add_special_tokens = True, return_tensors = 'pt')
        
        labels = summary_encoding['input_ids']
        labels[labels == 0] = -100
        
        return dict(text = text, summary = data_row['summary'], text_input_ids = text_encoding['input_ids'], text_attention_mask = text_encoding['attention_mask'].flatten(), labels = labels.flatten(), labels_attention_mask = summary_encoding['attention_mask'].flatten())
    
class NewsSummaryDataModule(pl.LightningDataModule):
    
    def __init__(self):
        
# hier weiter: https://www.youtube.com/watch?v=KMyZUIraHio&ab_channel=VenelinValkov        
