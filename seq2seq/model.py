import os
import argparse
import time

from icecream import ic
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoConfig, AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration, get_linear_schedule_with_warmup
import pytorch_lightning as pl

from dataset import SparcDataset
from evaluation import evaluate as evaluate_sparc


class SQLSeq2seqModel(pl.LightningModule):
    def __init__(self, config_name, config_dict, data_path='data/sparc', save_path='logdir/sparc'):
        super().__init__()
        self.config_name = config_name
        self.data_path = data_path
        self.save_path = save_path
        self.config_dict = config_dict
        self.tokenizer = AutoTokenizer.from_pretrained(config_name)
        if 't5' in config_name:
            self.model = T5ForConditionalGeneration.from_pretrained(config_name)
            self.tokenizer.add_tokens(['<'])
            print('Add \'<\' token into vocabulary.')
            self.model.resize_token_embeddings(len(self.tokenizer))
        elif 'bart' in config_name:
            self.model = BartForConditionalGeneration.from_pretrained(config_name)
        self.generate_interval = 1

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        model_output = self.model(**self.process_input_dict(x))
        lm_logits = model_output.logits
        masked_lm_loss = model_output.loss
        return {'lm_logits': lm_logits, 'loss': masked_lm_loss}

    def training_step(self, x, batch_idx):
        model_output = self.model(**self.process_input_dict(x))
        masked_lm_loss = model_output.loss
        self.log('train_loss', masked_lm_loss, sync_dist=True, prog_bar=True)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        input_dict = self.process_input_dict(x)
        model_output = self.model(**input_dict)
        masked_lm_loss = model_output.loss
        self.log('val_loss', masked_lm_loss, sync_dist=True, prog_bar=True)

        pred_lfs = []
        if self.current_epoch % self.generate_interval == 0:
            pred_ids = self.model.generate(input_ids=x['input_ids'], attention_mask=x['attention_mask'],
                                           num_beams=1, max_length=128, early_stopping=True, no_repeat_ngram_size=0)
            for i in range(x['id'].size(0)):
                pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])[1:]
                if self.tokenizer.eos_token in pred_lf:
                    pred_lf = pred_lf[:pred_lf.index(self.tokenizer.eos_token)]
                pred_lf = self.tokenizer.convert_tokens_to_string(pred_lf)
                pred_lfs.append((x['id'][i].item(), pred_lf))
        return {'pred_lfs': pred_lfs, 'loss': masked_lm_loss}

    def validation_step_end(self, step_output):
        pred_dict = {}
        if self.current_epoch % self.generate_interval == 0:
            for idx, pred_lf in step_output['pred_lfs']:
                pred_dict[idx] = pred_lf
            os.makedirs(os.path.join(self.save_path, 'predict'), exist_ok=True)
            with open(os.path.join(self.save_path, f'predict/predict_rank_{self.global_rank}.txt'), 'a') as fa:
                for idx, pred_lf in pred_dict.items():
                    fa.write(f'{idx}\t{pred_lf}\n')
        return pred_dict

    def validation_epoch_end(self, validation_step_output):
        if self.global_rank == 0 and self.current_epoch % self.generate_interval == 0:
            time.sleep(3)
            pred_dict = {}
            for i in range(8):
                if os.path.exists(os.path.join(self.save_path, f'predict/predict_rank_{i}.txt')):
                    with open(os.path.join(self.save_path, f'predict/predict_rank_{i}.txt')) as fr:
                        for line in fr:
                            idx, pred_lf = line.strip().split('\t')
                            pred_dict[int(idx)] = pred_lf
                    os.remove(os.path.join(self.save_path, f'predict/predict_rank_{i}.txt'))
            pred_list = sorted(pred_dict.items())
            with open(os.path.join(self.save_path, 'predict/predict.txt'), 'w') as fw:
                for idx, pred_lf in pred_list:
                    fw.write(f'{idx}\t{pred_lf}\n')
            exact_match_acc = evaluate_sparc(gold=os.path.join(self.data_path, 'dev_gold.txt'),
                                             predict=os.path.join(self.save_path, 'predict/predict.txt'),
                                             db_dir=os.path.join(self.data_path, 'database'),
                                             table=os.path.join(self.data_path, 'tables.json'),
                                             etype='match')
            # self.log('exact match', exact_match_acc, sync_dist=True, on_step=False, on_epoch=True)
            ic(exact_match_acc)

    def process_input_dict(self, x):
        d = {}
        for k in x:
            if k == 'id':
                continue
            else:
                d[k] = x[k]
        return d

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config_dict.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=self.config_dict.learning_rate, eps=self.config_dict.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    # def optimizer_step(self,
    #                    epoch,
    #                    batch_idx,
    #                    optimizer,
    #                    optimizer_idx,
    #                    optimizer_closure,
    #                    on_tpu=False,
    #                    using_native_amp=False,
    #                    using_lbfgs=False, ):
    #     optimizer.step()
    #     optimizer.zero_grad()
    #     self.lr_scheduler.step()

    def train_dataloader(self):
        train_dataset = SparcDataset(os.path.join(self.data_path, 'train.json'),
                                     os.path.join(self.data_path, 'tables.json'),
                                     os.path.join(self.data_path, 'database'),
                                     tokenizer=self.tokenizer)
        dataloader = DataLoader(train_dataset, batch_size=self.config_dict.train_batch_size, drop_last=True, shuffle=True, num_workers=4, collate_fn=train_dataset.collate_fn)
        t_total = (
                (len(dataloader.dataset) // (self.config_dict.train_batch_size * max(1, self.config_dict.n_gpu)))
                // self.config_dict.gradient_accumulation_steps
                * float(self.config_dict.num_train_epochs)
        )
        # scheduler = get_linear_schedule_with_warmup(
        #     self.opt, num_warmup_steps=self.config_dict.warmup_steps, num_training_steps=t_total
        # )
        # self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = SparcDataset(os.path.join(self.data_path, 'dev.json'),
                                   os.path.join(self.data_path, 'tables.json'),
                                   os.path.join(self.data_path, 'database'),
                                   tokenizer=self.tokenizer)
        return DataLoader(val_dataset, batch_size=self.config_dict.train_batch_size, num_workers=4, collate_fn=val_dataset.collate_fn)
