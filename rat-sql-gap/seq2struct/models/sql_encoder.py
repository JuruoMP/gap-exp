import torch
from torch.utils.data.dataloader import default_collate
from transformers import BertModel, BertTokenizer, BartModel, BartTokenizer

from seq2struct.utils import registry


@registry.register('sql_encoder', 'bert-encoder')
class SqlBertEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self._device = device
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self.bert = BertModel.from_pretrained('bert-base-cased').to(device)

    def forward(self, sql_list):
        input_dict = self.tokenizer(sql_list, padding=True)
        input_tensor_dict = {k: torch.LongTensor(v).to(self._device) for k, v in input_dict.items()}
        output = self.bert(**input_tensor_dict)
        return output


@registry.register('sql_encoder', 'bart-encoder')
class SqlBartEncoder(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self._device = device
        self.bart = BartModel.from_pretrained('facebook/bart-base').to(device)

    def forward(self, x):
        return x
