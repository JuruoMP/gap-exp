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

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        model_output = self.model(**self.filter_input_dict(x))
        lm_logits = model_output.logits
        masked_lm_loss = model_output.loss
        return {'lm_logits': lm_logits, 'loss': masked_lm_loss}

    def training_step(self, x, batch_idx):
        model_output = self.model(**self.filter_input_dict(x))
        masked_lm_loss = model_output.loss
        self.log('train_loss', masked_lm_loss, sync_dist=True, prog_bar=True)
        return masked_lm_loss

    def validation_step(self, x, batch_idx):
        model_output = self.model(**self.filter_input_dict(x))
        masked_lm_loss = model_output.loss
        self.log('val_loss', masked_lm_loss, sync_dist=True, prog_bar=True)

        pred_lfs = []
        pred_ids = self.model.generate(x['input_ids'], num_beams=1, max_length=512, early_stopping=True, no_repeat_ngram_size=0)
        for i in range(x['id'].size(0)):
            pred_lf = self.tokenizer.convert_ids_to_tokens(pred_ids[i])[1:]
            if self.tokenizer.eos_token in pred_lf:
                pred_lf = pred_lf[:pred_lf.index(self.tokenizer.eos_token)]
            pred_lf = self.tokenizer.convert_tokens_to_string(pred_lf)
            pred_lfs.append((x['id'][i].item(), pred_lf))
        return {'pred_lfs': pred_lfs, 'loss': masked_lm_loss}

    def validation_step_end(self, step_output):
        pred_dict = {}
        for idx, pred_lf in step_output['pred_lfs']:
            pred_dict[idx] = pred_lf
        os.makedirs(os.path.join(self.save_path, 'predict'), exist_ok=True)
        with open(os.path.join(self.save_path, f'predict/predict_rank_{self.global_rank}.txt', 'a')) as fa:
            for idx, pred_lf in pred_dict.items():
                fa.write(f'{idx}\t{pred_lf}\n')
        return pred_dict

    def validation_epoch_end(self, validation_step_output):
        if self.global_rank == 0:
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

    def filter_input_dict(self, x):
        d = {}
        for k in x:
            if k not in ('id'):
                d[k] = x[k]
        return d

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return optimizer