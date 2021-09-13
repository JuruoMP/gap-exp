import os
import sys
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import random
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model import SQLSeq2seqModel
from dataset import SparcDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(1)


if __name__ == '__main__':
    # config_name = 'facebook/bart-base'
    config_name = 't5-large'
    # train_dataset = SparcDataset('data/sparc/train.json', 'data/sparc/tables.json', 'data/sparc/database', config_name=config_name)
    # dev_dataset = SparcDataset('data/sparc/dev.json', 'data/sparc/tables.json', 'data/sparc/database', config_name=config_name)
    # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, collate_fn=train_dataset.collate_fn)
    # dev_dataloader = DataLoader(dev_dataset, batch_size=2, shuffle=False, collate_fn=dev_dataset.collate_fn)

    config_dict = argparse.Namespace(
        learning_rate=3e-4,
        weight_decay=0.0,
        adam_epsilon=1e-8,
        warmup_steps=0,
        train_batch_size=8,
        n_gpu=torch.cuda.device_count(),
        gradient_accumulation_steps=1,
        num_train_epochs=100,
    )
    model = SQLSeq2seqModel(config_name=config_name, config_dict=config_dict)
    # trainer = pl.Trainer(gpus=0, default_root_dir=f'logdir/{config_name}',
    #                      terminate_on_nan=True,
    #                      gradient_clip_val=0.5,
    #                      callbacks=[EarlyStopping(monitor='val_loss', patience=10, mode='min')])
    trainer = pl.Trainer(gpus=-1, default_root_dir=f'logdir/{config_name}',
                         terminate_on_nan=True, accelerator='ddp', precision=16, plugins="deepspeed_stage_3",
                         gradient_clip_val=0.5, max_epochs=100,
                         callbacks=[EarlyStopping(monitor='val_loss', patience=3, mode='min')])
    trainer.fit(model)

    # trainer.test(test_dataloaders=dev_dataloader)
