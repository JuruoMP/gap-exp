import torch
import torch.utils.data

from seq2struct.models import abstract_preproc
from seq2struct.utils import registry
from global_config import global_config

import numpy as np

class ZippedDataset(torch.utils.data.Dataset):
    def __init__(self, *components):
        assert len(components) >= 1
        lengths = [len(c) for c in components]
        assert all(
            lengths[0] == other for other in lengths[1:]), "Lengths don't match: {}".format(lengths)
        self.components = components
    
    def __getitem__(self, idx):
        return tuple(c[idx] for c in self.components)
    
    def __len__(self):
        return len(self.components[0])


@registry.register('model', 'EncDec')
class EncDecModel(torch.nn.Module):
    class Preproc(abstract_preproc.AbstractPreproc):
        def __init__(
                self,
                encoder,
                decoder,
                encoder_preproc,
                decoder_preproc):
            super().__init__()

            self.enc_preproc = registry.lookup('encoder', encoder['name']).Preproc(**encoder_preproc)
            self.dec_preproc = registry.lookup('decoder', decoder['name']).Preproc(**decoder_preproc)
        
        def validate_item(self, item, section):
            enc_result, enc_info = self.enc_preproc.validate_item(item, section)
            dec_result, dec_info = self.dec_preproc.validate_item(item, section)
            
            return enc_result and dec_result, (enc_info, dec_info)
        
        def add_item(self, item, section, validation_info):
            enc_info, dec_info = validation_info
            self.enc_preproc.add_item(item, section, enc_info)
            self.dec_preproc.add_item(item, section, dec_info)
        
        def clear_items(self):
            self.enc_preproc.clear_items()
            self.dec_preproc.clear_items()

        def save(self):
            self.enc_preproc.save()
            self.dec_preproc.save()
        
        def load(self):
            self.enc_preproc.load()
            self.dec_preproc.load()
        
        def dataset(self, section):
            return ZippedDataset(self.enc_preproc.dataset(section), self.dec_preproc.dataset(section))
        
    def __init__(self, preproc, device, encoder, decoder):
        super().__init__()
        self.preproc = preproc
        self.encoder = registry.construct(
                'encoder', encoder, device=device, preproc=preproc.enc_preproc)
        self.decoder = registry.construct(
                'decoder', decoder, device=device, preproc=preproc.dec_preproc)
        self.decoder.visualize_flag = False

        # self.sql_encoder = registry.construct('sql_encoder', {'name': 'bert-encoder'}, device=device)
        # self.sql_ff = torch.nn.Linear(768, 1024)
        self.sql_cache = {}
        
        if getattr(self.encoder, 'batched'):
            self.compute_loss = self._compute_loss_enc_batched
        else:
            self.compute_loss = self._compute_loss_unbatched

    def _compute_loss_enc_batched(self, batch, debug=False):
        losses = []
        d = [enc_input for enc_input, dec_output in batch]
        enc_states = self.encoder(d)

        reg_loss1 = []
        reg_loss2 = []
        tc_loss = []
        contrast_loss = []

        is_history = False
        for enc_state, (enc_input, dec_output) in zip(enc_states, batch):
            _loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)

            sql_str = dec_output.sql_str
            sql_contrast_loss = None
            if sql_str in self.sql_cache:
                target = self.sql_cache.get(sql_str)
                sql_contrast_loss = torch.nn.functional.mse_loss(target, enc_state.bert_output, reduction='mean')
            self.sql_cache[sql_str] = enc_state.bert_output.detach()

            if isinstance(_loss, tuple):
                is_history = True
                _loss_tensor = _loss[0]
                if global_config.contrast_loss_weight > 0 and sql_contrast_loss is not None:
                    _loss_tensor += global_config.contrast_loss_weight * sql_contrast_loss
                    contrast_loss.append(sql_contrast_loss.item())
                losses.append(_loss_tensor)
                reg_loss1.append(_loss[1][0].item())
                reg_loss2.append(_loss[1][1].item())
                if enc_state.tc_loss != 0.0:
                    tc_loss.append(enc_state.tc_loss.item())
            else:
                if global_config.contrast_loss_weight > 0 and sql_contrast_loss is not None:
                    _loss += global_config.contrast_loss_weight * sql_contrast_loss
                losses.append(_loss)
        if len(tc_loss) == 0:
            tc_loss = [0.0]

        # use sql_encoder to calculate contrastive loss
        # contrast_loss = []
        # if True:  # contrastive loss
        #     sql_str_list = [dec_output.sql_str for enc_input, dec_output in batch]
        #     sql_reprs = self.sql_ff(self.sql_encoder(sql_str_list).pooler_output)
        #     cls_vecs = torch.stack([x.bert_output for x in enc_states], dim=0)
        #     contrast_loss.append(torch.nn.functional.mse_loss(cls_vecs, sql_reprs, reduction='mean'))

        if debug:
            return losses
        else:
            ret = (torch.mean(torch.stack(losses, dim=0), dim=0),)
            if is_history:
                ret = ret + ((np.mean(reg_loss1), np.mean(reg_loss2)), np.mean(tc_loss))
                if contrast_loss:
                    ret = ret + (np.mean(contrast_loss), )
                else:
                    ret = ret + (None, )
        return ret

    def _compute_loss_enc_batched2(self, batch, debug=False):
        losses = []
        for enc_input, dec_output in batch:
            enc_state, = self.encoder([enc_input])
            loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            losses.append(loss)
        if debug:
            return losses
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def _compute_loss_unbatched(self, batch, debug=False):
        losses = []
        reg_loss1 = []
        reg_loss2 = []
        contrast_loss = []
        is_history = False
        for enc_input, dec_output in batch:
            enc_state = self.encoder(enc_input)
            _loss = self.decoder.compute_loss(enc_input, dec_output, enc_state, debug)
            if isinstance(_loss, tuple):
                is_history = True
                losses.append(_loss[0])
                reg_loss1.append(_loss[1][0].item())
                reg_loss2.append(_loss[1][1].item())
                if _loss[2]:
                    contrast_loss.append(_loss[2])
            else:
                losses.append(_loss)

        if debug:
            return losses
        elif is_history:
            return torch.mean(torch.stack(losses, dim=0), dim=0), (np.mean(reg_loss1), np.mean(reg_loss2))
        else:
            return torch.mean(torch.stack(losses, dim=0), dim=0)

    def eval_on_batch(self, batch):
        _loss = self.compute_loss(batch)
        if isinstance(_loss,tuple):
            mean_loss = _loss[0].item()
            batch_size = len(batch)
            result = {'loss': mean_loss * batch_size,
                      'reg_1': _loss[1][0] * batch_size, 'reg_2': _loss[1][1] * batch_size,
                      'tc': _loss[2] * batch_size,
                      'contrast': _loss[3] if len(_loss) > 3 else None,
                      'total': batch_size}
            return result
        else:
            mean_loss = self.compute_loss(batch).item()
            batch_size = len(batch)
            result = {'loss': mean_loss * batch_size, 'total': batch_size}
            return result

    def begin_inference(self, orig_item, preproc_item):
        ## TODO: Don't hardcode train
        #valid, validation_info =  self.preproc.enc_preproc.validate_item(item, 'train')
        #if not valid:
        #    return None
        #enc_input = self.preproc.enc_preproc.preprocess_item(item, validation_info)

        enc_input, _ = preproc_item
        if self.decoder.visualize_flag:
            print('question:')
            print(enc_input['question'])
            print('columns:')
            print(enc_input['columns'])
            print('tables:')
            print(enc_input['tables'])
        if getattr(self.encoder, 'batched'):
            enc_state, = self.encoder([enc_input])
        else:
            enc_state = self.encoder(enc_input)
        return self.decoder.begin_inference(enc_state, orig_item)
