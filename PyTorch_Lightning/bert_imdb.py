import torch
from pytorch import Dataset
import pytorch_lightning as pl

import transformers

# https://medium.com/@knswamy/sequence-classification-using-pytorch-lightning-with-bert-on-imbd-data-5e9f48baa638

# custom dataset uses Bert Tokenizer to create the Pytorch Dataset
class ImdbDataset(Dataset):
    def __init__(self, notes, targets, tokenizer, max_len):
        self.notes = notes
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_len = max_len
         
    def __len__(self):
        return (len(self.notes))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        note = str(self.notes[idx])
        target = self.targets[idx]
        
        encoding = self.tokenizer.encode_plus(
          note,
          add_special_tokens=True,
          max_length=self.max_len,
          return_token_type_ids=True,
          truncation=True,
          padding='max_length',
          return_attention_mask=True,
          return_tensors='pt',
        )    
        return {
            #'text': note,
            'label': torch.tensor(target, dtype=torch.long),
            'input_ids': (encoding['input_ids']).flatten(),
            'attention_mask': (encoding['attention_mask']).flatten(),
            'token_type_ids': (encoding['token_type_ids']).flatten()
        }

## The main Pytorch Lightning module
class ImdbModel(pl.LightningModule):

    def __init__(self,
                 learning_rate: float = 0.0001 * 8,
                 batch_size: int = 64,
                 num_workers: int = 4,
                 **kwargs):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.num_labels = 2
        config = transformers.DistilBertConfig(dropout=0.2, attention_dropout=0.2)
        self.bert = transformers.DistilBertModel.from_pretrained('distilbert-base-uncased', config=config)
        
        self.pre_classifier = torch.nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size)
        self.classifier = torch.nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.dropout = torch.nn.Dropout(self.bert.config.seq_classif_dropout)

        # relu activation function
        self.relu =  torch.nn.ReLU()

    
    def forward(self, input_ids, attention_mask, labels):
      
        outputs = self.bert(input_ids=input_ids, \
                         attention_mask=attention_mask)

        hidden_state = outputs[0]  # (bs, seq_len, dim)
        pooled_output = hidden_state[:, 0]  # (bs, dim)
        pooled_output = self.pre_classifier(pooled_output)  # (bs, dim)
        pooled_output = self.relu(pooled_output)  # (bs, dim)
        pooled_output = self.dropout(pooled_output)  # (bs, dim)
        logits = self.classifier(pooled_output)  # (bs, dim)

        return logits
    
    def training_step(self, batch, batch_nb):
        # batch
        input_ids = batch['input_ids']
        label = batch['label']
        attention_mask = batch['attention_mask']
        #token_type_ids = batch['token_type_ids']
        # fwd
        y_hat = self(input_ids, attention_mask, label)
        
        # loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(y_hat.view(-1, self.num_labels), label.view(-1))
        #loss = F.cross_entropy(y_hat, label)
        
        # logs
        tensorboard_logs = {'train_loss': loss, 'learn_rate': self.optim.param_groups[0]['lr'] }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def on_batch_end(self):
    # This is needed to use the One Cycle learning rate that needs the learning rate to change after every batch
    # Without this, the learning rate will only change after every epoch
        if self.sched is not None:
            self.sched.step()


def run_cli():
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    #root_dir = os.path.dirname(os.path.realpath(__file__))
    root_dir = os. getcwd()
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    
    # each LightningModule defines arguments relevant to it
    parser = ImdbModel.add_model_specific_args(parent_parser,root_dir)
    
    parser.set_defaults(
        profiler=True,
        deterministic=True,
        max_epochs=1,
        gpus=1,
        distributed_backend=None,
        fast_dev_run=False,
        model_load=False,
        model_name='best_model',
    )
    
    #args = parser.parse_args()
    args, extra = parser.parse_known_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(args)

