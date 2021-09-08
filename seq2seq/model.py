import os

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoTokenizer, BartForConditionalGeneration, T5ForConditionalGeneration
import pytorch_lightning as pl

from evaluation import evaluate as evaluate_sparc


class SQLSeq2seqModel(pl.LightningModule):
    def __init__(self, config_name, data_path='data/sparc', save_path='logdir'):
        super().__init__()
        self.config_name = config_name
        self.data_path = data_path
        self.save_path = save_path
        self.tokenizer = AutoTokenizer.from_pretrained(config_name)
        if 't5' in config_name:
            self.model = T5ForConditionalGeneration.from_pretrained(config_name)
        elif 'bart' in config_name:
            self.model = BartForConditionalGeneration.from_pretrained(config_name)
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        lm_logits = self.model(**self.filter_input_dict(x)).logits
        pred_ids = lm_logits.argmax(dim=-1)
        pred_str = [self.tokenizer.convert_ids_to_tokens(xx) for xx in pred_ids]
        masked_lm_loss = self.loss_fct(lm_logits.flatten(0, -2), x['labels'].view(-1))
        return {'lm_logits': lm_logits, 'sql': pred_str, 'loss': masked_lm_loss}

    def training_step(self, x, batch_idx):
        lm_logits = self.model(**self.filter_input_dict(x)).logits
        masked_lm_loss = self.loss_fct(lm_logits.flatten(0, -2), x['labels'].view(-1))

        self.log('train_loss', masked_lm_loss)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        lm_logits = self.model(**self.filter_input_dict(x)).logits
        masked_lm_loss = self.loss_fct(lm_logits.flatten(0, -2), x['labels'].view(-1))
        self.log('val_loss', masked_lm_loss)
        pred_ids = lm_logits.argmax(dim=-1)
        pred_lfs = []
        for i in range(pred_ids.size(0)):
            pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])
            pred_lfs.append((x['id'][i].item(), pred_lf))
        return (pred_lfs, masked_lm_loss)

    def validation_epoch_end(self, val_ret):
        if self.global_rank == 0:
            pred_list = [j for i in val_ret for j in i[0]]
            losses = [i[1].item() for i in val_ret]
            avg_loss = sum(losses) / len(losses)
            new_pred_list = []
            for i in range(len(pred_list)):
                idx, pred = pred_list[i]
                if self.tokenizer.eos_token in pred:
                    pred = pred[1:pred.index(self.tokenizer.eos_token)]
                else:
                    pred = pred[1:]
                pred = self.tokenizer.convert_tokens_to_string(pred)
                new_pred_list.append((idx, pred))
            if not os.path.exists(os.path.join(self.save_path, 'predict')):
                os.makedirs(os.path.join(self.save_path, 'predict'))
            new_pred_list = sorted(new_pred_list, key=lambda x: x[0])
            with open(os.path.join(self.save_path, 'predict/predict.txt'), 'w') as fw:
                for idx, pred in new_pred_list:
                    fw.write(pred + '\n')
            if self.current_epoch % 10 == 0:
                with open(os.path.join(self.save_path, f'predict/predict_{self.current_epoch}.txt'), 'w') as fw:
                    for idx, pred in new_pred_list:
                        fw.write(pred + '\n')
            exact_match_acc = evaluate_sparc(
                os.path.join(self.data_path, 'dev_gold.txt'), os.path.join(self.save_path, 'predict/predict.txt'),
                os.path.join(self.data_path, 'database'), os.path.join(self.data_path, 'tables.json'))
            self.log('val_acc', exact_match_acc)
            print(f'Validation exact match acc = {exact_match_acc:.3f}, loss = {avg_loss:.3e}')

    def filter_input_dict(self, x):
        d = {}
        for k in x:
            if k not in ('id'):
                d[k] = x[k]
        return d

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer