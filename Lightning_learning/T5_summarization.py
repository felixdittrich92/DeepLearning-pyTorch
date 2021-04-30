import os
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

from transformers import AdamW, T5ForConditionalGeneration, T5TokenizerFast
from tqdm.auto import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "true"

df = pd.read_csv("/home/felix/Desktop/DeepLearning-pyTorch/Lightning_learning/data/news_summary.csv", encoding="latin-1")
df = df[["text", "ctext"]]
df.columns = ["summary", "text"]
print(df.head())
df = df.dropna()

train_df, test_df = train_test_split(df, test_size=0.15)

class NewsSummaryDataset(Dataset):
    
    def __init__(self, data: pd.DataFrame, tokenizer: T5TokenizerFast, text_max_token_len = 512, summary_max_token_len = 128):
        self.tokenizer = tokenizer
        self.data = data
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        text = data_row["text"]
        
        text_encoding = self.tokenizer(text, 
                                       max_length=self.text_max_token_len, 
                                       padding='max_length', 
                                       truncation=True,
                                       return_attention_mask=True,
                                       add_special_tokens=True, 
                                       return_tensors='pt'
                                       )
        
        summary_encoding = self.tokenizer(text, 
                                          max_length=self.summary_max_token_len, 
                                          padding='max_length', 
                                          truncation=True, 
                                          return_attention_mask=True, 
                                          add_special_tokens=True, 
                                          return_tensors='pt'
                                          )
        
        labels = summary_encoding["input_ids"]
        labels[labels == 0] = -100
        
        return dict(text=text, 
                    summary=data_row["summary"], 
                    text_input_ids=text_encoding["input_ids"], 
                    text_attention_mask=text_encoding["attention_mask"].flatten(), 
                    labels=labels.flatten(), 
                    labels_attention_mask=summary_encoding["attention_mask"].flatten()
                    )
    
class NewsSummaryDataModule(pl.LightningDataModule):
    
    def __init__(self, train_df: pd.DataFrame, test_df: pd.DataFrame, tokenizer: T5TokenizerFast, batch_size: int = 8, text_max_token_len: int = 512, summary_max_token_len: int = 128):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.text_max_token_len = text_max_token_len
        self.summary_max_token_len = summary_max_token_len
    
    def setup(self, stage=None):
        self.train_dataset = NewsSummaryDataset(
            self.train_df, 
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )
        
        self.test_dataset = NewsSummaryDataset(
            self.test_df, 
            self.tokenizer,
            self.text_max_token_len,
            self.summary_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=8
        )
        
MODEL_NAME = 't5-base'
tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)

text_token_counts, summary_token_counts = [], []

for _, row in train_df.iterrows():
    text_token_count = len(tokenizer.encode(row["text"]))
    text_token_counts.append(text_token_count)
    
    summary_token_count = len(tokenizer.encode(row["summary"]))
    summary_token_counts.append(summary_token_count)

print('Text Tokens :' + str(sum(text_token_counts)))
print('Summary Tokens :' + str(sum(summary_token_counts)))

N_EPOCHS = 3
BATCH_SIZE = 8

data_module = NewsSummaryDataModule(train_df, test_df, tokenizer, batch_size=BATCH_SIZE)

# MODEL

class NewsSummaryModel(pl.LightningModule):
    
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, return_dict=True)
        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None):
        loss, logits = self.model(input_ids, 
                                  attention_mask=attention_mask,
                                  labels=labels,
                                  decoder_attention_mask=decoder_attention_mask
                                  )

        return loss, logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
            )
        self.log("train_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
            )
        self.log("val_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        input_ids = batch["text_input_ids"]
        attention_mask = batch["text_attention_mask"]
        labels = batch["labels"]
        labels_attention_mask = batch["labels_attention_mask"]
        
        loss, outputs = self(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_attention_mask=labels_attention_mask,
            labels=labels
            )
        self.log("test_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=0.0001)
    
model = NewsSummaryModel()
      
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger(save_dir="Lightning_logs", name="news_summary")

early_stopping_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.01,
    patience=2,
    verbose=True
)

trainer = pl.Trainer(logger=logger,
                     callbacks=[checkpoint_callback, early_stopping_callback],
                     max_epochs=N_EPOCHS,
                     gpus=0,
                     progress_bar_refresh_rate=1
                     )

trainer.fit(model, data_module)

trained_model = NewsSummaryModel.load_from_checkpoint(
    trainer.checkpoint_callback.best_model_path
)
trained_model.freeze()

def summarize(text):
    text_encoding = tokenizer(
        text, 
        max_length=512, 
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )
    
    generated_ids = trained_model.model.generate(
        input_ids=text_encoding["input_ids"],
        attention_mask=text_encoding["attention_mask"],
        max_length=150,
        num_beams=2,
        repetition_penalty=2.5,
        length_penalty=1.0,
        early_stopping=True
    )
    
    preds = [
        tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for gen_id in generated_ids
    ]
    
    return "".join(preds)

sample_row = test_df.iloc[0]
text = sample_row["text"]
model_summary = summarize(text)

print(model_summary)
    

    