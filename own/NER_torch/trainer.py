import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from seqeval.metrics import classification_report as seqeval_classification

from ner_dataset import unique_tags, index2tag, train, val, test
from torch_model import NerBertModel, eval_labels, eval_preds


model = NerBertModel(unique_tags=unique_tags)
model_parameter = { "names": [name for name, param in list(model.named_parameters())], 
                    "param": [param for name, param in list(model.named_parameters())]}
print(f'Model parameter_names : \n {model_parameter["names"]} \n')

      
checkpoint_callback = ModelCheckpoint(
    dirpath="checkpoints",
    filename="best-checkpoint",
    save_top_k=1,
    verbose=True,
    monitor="val_loss",
    mode="min"
)

logger = TensorBoardLogger(save_dir="Lightning_logs", name="ConLL")

early_stopping_callback = EarlyStopping(
    monitor="val_loss", 
    min_delta=0.01,
    patience=2,
    verbose=True
)

trainer = pl.Trainer(logger=logger,
                     callbacks=[checkpoint_callback, early_stopping_callback],
                     max_epochs=100,
                     gpus=0, # change to 1 
                     auto_lr_find=True,
                   #  precision=16, only with gpu
                     progress_bar_refresh_rate=1
                     )

trainer.fit(model, train, val)
trainer.test(model, test)

all_labels = [index2tag[ids.item()] for ids in eval_labels]
all_predictions = [index2tag[ids.item()] for ids in eval_preds]
print('----------SKLEARN REPORT----------')
print(classification_report(all_labels, all_predictions))
print('----------SEQEVAL REPORT----------')
print(seqeval_classification(all_labels, all_predictions))

trained_model = model.load_from_checkpoint(
  trainer.checkpoint_callback.best_model_path,
  unique_tags=unique_tags
)
trained_model.eval()
trained_model.freeze()


# ----------------------------------
# torchscript
# ----------------------------------
import torch
torch.jit.save(trained_model.to_torchscript(), "model/ner_model.pt")
if os.path.isfile("model/ner_model.pt"):
  print('saved model in torchscript format')

# ----------------------------------
# onnx
# ----------------------------------
"""
from tempfile import NamedTemporaryFile
with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmpfile:
     input_sample = torch.randn((1, 28 * 28)) # result = tokenizer.encode_plus() ....
     trained_model.to_onnx(tmpfile.name, input_sample, export_params=True)
     if os.path.isfile(tmpfile.name):
       print('saved model in ONNX format')
"""


# https://huggingface.co/transformers/serialization.html
# TODO:  ONNX Export / TORCH Export / work on Inference
# TODO: save pretrained tokenizer and bert ??


