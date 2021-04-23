import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm.notebook import tqdm

torch.manual_seed(2020)

print(torch.cuda.get_device_name(torch.cuda.current_device()))
print(torch.cuda.is_available())
print(torch.__version__)

import pandas as pd
import numpy as np

data = pd.read_csv("PyTorch_Lightning/ner_dataset.csv", encoding="latin1")
data = data.fillna(method="ffill")

class SentenceGetter(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False
        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None
        

getter = SentenceGetter(data)
sentences = getter.sentences

tags = ["[PAD]"]
tags.extend(list(set(data["Tag"].values)))
tag2idx = {t: i for i, t in enumerate(tags)}

words = ["[PAD]", "[UNK]"]
words.extend(list(set(data["Word"].values)))
word2idx = {t: i for i, t in enumerate(words)}

test_sentences, val_sentences, train_sentences = sentences[:15000], sentences[15000:20000], sentences[20000:]

import random
from transformers import pipeline

class TransformerAugmenter():
    """
    Use the pretrained masked language model to generate more
    labeled samples from one labeled sentence.
    """
    
    def __init__(self):
        self.num_sample_tokens = 5
        self.fill_mask = pipeline(
            "fill-mask",
            topk=self.num_sample_tokens,
            model="distilroberta-base"
        )
    
    def generate(self, sentence, num_replace_tokens=3):
        """Return a list of n augmented sentences."""
              
        # run as often as tokens should be replaced
        augmented_sentence = sentence.copy()
        for i in range(num_replace_tokens):
            # join the text
            text = " ".join([w[0] for w in augmented_sentence])
            # pick a token
            replace_token = random.choice(augmented_sentence)
            # mask the picked token
            masked_text = text.replace(
                replace_token[0],
                f"{self.fill_mask.tokenizer.mask_token}",
                1            
            )
            # fill in the masked token with Bert
            res = self.fill_mask(masked_text)[random.choice(range(self.num_sample_tokens))]
            # create output samples list
            tmp_sentence, augmented_sentence = augmented_sentence.copy(), []
            for w in tmp_sentence:
                if w[0] == replace_token[0]:
                    augmented_sentence.append((res["token_str"].replace("Ġ", ""), w[1], w[2]))
                else:
                    augmented_sentence.append(w)
            text = " ".join([w[0] for w in augmented_sentence])
        return [sentence, augmented_sentence]

augmenter = TransformerAugmenter()

augmented_sentences = augmenter.generate(train_sentences[12], num_replace_tokens=7)

# only use a thousand senteces with augmentation
n_sentences = 1000

augmented_sentences = []
for sentence in tqdm(train_sentences[:n_sentences]):
    augmented_sentences.extend(augmenter.generate(sentence, num_replace_tokens=7))
    
print(len(augmented_sentences))

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy, f1_score

from keras.preprocessing.sequence import pad_sequences

EMBEDDING_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 64
MAX_LEN = 50

class LightningLSTMTagger(pl.LightningModule):

    def __init__(self, embedding_dim, hidden_dim):
        super(LightningLSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(len(word2idx), embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, len(tag2idx))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = lstm_out
        logits = self.fc(lstm_out)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.permute(0, 2, 1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        result = pl.TrainResult(minimize=loss)
        result.log('f1', f1_score(torch.argmax(y_hat, dim=1), y), prog_bar=True)
        return result
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.permute(0, 2, 1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        result = pl.EvalResult()
        result.log('val_f1', f1_score(torch.argmax(y_hat, dim=1), y), prog_bar=True)
        return result
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y_hat = y_hat.permute(0, 2, 1)
        loss = nn.CrossEntropyLoss()(y_hat, y)
        return {'test_f1':  f1_score(torch.argmax(y_hat, dim=1), y)}
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=5e-4)
    
def get_dataloader(seqs, max_len, batch_size, shuffle=False):
    input_ids = pad_sequences([[word2idx.get(w[0], word2idx["[UNK]"]) for w in sent] for sent in seqs],
                              maxlen=max_len, dtype="long", value=word2idx["[PAD]"],
                              truncating="post", padding="post")

    tag_ids = pad_sequences([[tag2idx[w[2]] for w in sent] for sent in seqs],
                              maxlen=max_len, dtype="long", value=tag2idx["[PAD]"],
                              truncating="post", padding="post")
    
    inputs = torch.tensor(input_ids)
    tags = torch.tensor(tag_ids)
    data = TensorDataset(inputs, tags)
    return DataLoader(data, batch_size=batch_size, num_workers=16, shuffle=shuffle)

ner_train_ds = get_dataloader(train_sentences[:2*n_sentences], MAX_LEN, BATCH_SIZE, shuffle=True)
ner_aug_train_ds = get_dataloader(augmented_sentences, MAX_LEN, BATCH_SIZE, shuffle=True)
ner_valid_ds = get_dataloader(val_sentences, MAX_LEN, BATCH_SIZE)
ner_test_ds = get_dataloader(test_sentences, MAX_LEN, BATCH_SIZE)

tagger = LightningLSTMTagger(
    EMBEDDING_DIM,
    HIDDEN_DIM
)

trainer = pl.Trainer(
    max_epochs=30,
    gradient_clip_val=100
)

trainings_results = trainer.fit(
    model=tagger,
    train_dataloader=ner_train_ds,
    val_dataloaders=ner_valid_ds
)

test_res = trainer.test(model=tagger, test_dataloaders=ner_test_ds, verbose=0)
print("Test F1-Score: {:.1%}".format(np.mean([res["test_f1"] for res in test_res])))

tagger = LightningLSTMTagger(
    EMBEDDING_DIM,
    HIDDEN_DIM
)

trainer = pl.Trainer(
    max_epochs=30,
    gradient_clip_val=100
)

trainer.fit(
    model=tagger,
    train_dataloader=ner_aug_train_ds,
    val_dataloaders=ner_valid_ds
)

test_res = trainer.test(model=tagger, test_dataloaders=ner_test_ds, verbose=0)
print("Test F1-Score: {:.1%}".format(np.mean([res["test_f1"] for res in test_res])))

