import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import SQLSeq2seqModel
from dataset import SparcDataset


if __name__ == '__main__':
    # config_name = 'facebook/bart-base'
    config_name = 't5-large'
    train_dataset = SparcDataset('data/sparc/train.json', 'data/sparc/tables.json', 'data/sparc/database', config_name=config_name)
    dev_dataset = SparcDataset('data/sparc/dev.json', 'data/sparc/tables.json', 'data/sparc/database', config_name=config_name)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn)
    dev_dataloader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=dev_dataset.collate_fn)

    model = SQLSeq2seqModel(config_name=config_name)
    # trainer = pl.Trainer(gpus=0, default_root_dir=f'logdir/{config_name}',
    #                      terminate_on_nan=True,
    #                      gradient_clip_val=5, gradient_clip_algorithm='value',
    #                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')])
    trainer = pl.Trainer(gpus=-1, precision=16, default_root_dir=f'logdir/{config_name}',
                         terminate_on_nan=True, accelerator='ddp', plugins="deepspeed_stage_2_offload",
                         gradient_clip_val=5, gradient_clip_algorithm='value',
                         callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')])
    trainer.fit(model, train_dataloader, dev_dataloader)

    trainer.test(test_dataloaders=dev_dataloader)
