import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl
from torchmetrics.functional import accuracy, f1

from transformers import BertForTokenClassification, AdamW

# lists for classification report
eval_labels = list()
eval_preds = list()


class NerBertModel(pl.LightningModule):
    def __init__(self, unique_tags: int):
        super(NerBertModel, self).__init__()
        self.num_labels = len(unique_tags)
        self.bert = BertForTokenClassification.from_pretrained('bert-base-german-cased', 
                                                               num_labels=self.num_labels, 
                                                               output_attentions=True, 
                                                               return_dict=True) 

    def forward(self, ids, mask, labels):
        outputs = self.bert(input_ids=ids, attention_mask=mask, labels=labels)
        # outputs.loss   outputs.logits   outputs.attentions
        return outputs.loss, outputs.logits

    
    def training_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        labels = batch["tags"]
        
        loss, logits = self(ids=ids, mask=mask, labels=labels)
        
        # compute accuracy
        flattened_targets = labels.view(-1) # shape (batch_size * seq_len,)
        active_logits = logits.view(-1, self.bert.num_labels) # shape (batch_size * seq_len, num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1) # shape (batch_size * seq_len,)
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 # shape (batch_size, seq_len)
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        return {"loss": loss, "logits": logits, "predictions": predictions, "labels": labels, "train_acc": accuracy(predictions, labels)}

    
    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_train_acc = torch.stack([x["train_acc"] for x in outputs]).mean()

        self.log("train_loss", avg_train_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("train_acc", avg_train_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)


    def validation_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        labels = batch["tags"]
        
        loss, logits = self(ids=ids, mask=mask, labels=labels)
        
        # compute accuracy
        flattened_targets = labels.view(-1) 
        active_logits = logits.view(-1, self.bert.num_labels) 
        flattened_predictions = torch.argmax(active_logits, axis=1) 
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
    
        return {"loss": loss, "logits": logits, "predictions": predictions, "labels": labels, "val_acc": accuracy(predictions, labels), "val_f1": f1(predictions, labels, num_classes=self.bert.num_labels)}
    
    def validation_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_val_acc = torch.stack([x["val_acc"] for x in outputs]).mean()
        avg_val_f1 = torch.stack([x["val_f1"] for x in outputs]).mean()
        
        self.log("val_loss", avg_val_loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_acc", avg_val_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("val_f1", avg_val_f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
        
    def test_step(self, batch, batch_idx):
        ids = batch["ids"]
        mask = batch["mask"]
        labels = batch["tags"]
        
        loss, logits = self(ids=ids, mask=mask, labels=labels)
        
        # compute accuracy
        flattened_targets = labels.view(-1) 
        active_logits = logits.view(-1, self.bert.num_labels) 
        flattened_predictions = torch.argmax(active_logits, axis=1) 
        
        # only compute accuracy at active labels
        active_accuracy = labels.view(-1) != -100 
        labels = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)
        
        # add for classification report
        eval_labels.extend(labels)
        eval_preds.extend(predictions)
    
        return {"loss": loss, "logits": logits, "predictions": predictions, "labels": labels, "test_acc": accuracy(predictions, labels), "test_f1": f1(predictions, labels, num_classes=self.bert.num_labels)}
    
    def test_epoch_end(self, outputs):
        avg_test_acc = torch.stack([x["test_acc"] for x in outputs]).mean()
        avg_test_f1 = torch.stack([x["test_f1"] for x in outputs]).mean()
        
        self.log("test_acc", avg_test_acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log("test_f1", avg_test_f1, prog_bar=True, logger=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=0.0001)
        scheduler = ReduceLROnPlateau(optimizer)
        self.log("learning_rate", optimizer.param_groups[0]['lr'], prog_bar=True, logger=True)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler,
            'monitor': 'val_loss'
        }