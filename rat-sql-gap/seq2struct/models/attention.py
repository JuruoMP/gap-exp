import math
import numpy as np
import torch
import torch.nn.functional as F

from seq2struct.utils import registry
from seq2struct.models import transformer


def maybe_mask(attn, attn_mask):
    if attn_mask is not None:
        assert all(
            a == 1 or b == 1 or a == b
             for a, b in zip(attn.shape[::-1], attn_mask.shape[::-1])), \
            'Attention mask shape {} should be broadcastable with attention shape {}'.format(
                attn_mask.shape, attn.shape)

        attn.data.masked_fill_(attn_mask, -float('inf'))


class Attention(torch.nn.Module):
    def __init__(self, pointer):
        super().__init__()
        self.pointer = pointer
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, query, values, attn_mask=None):
        # query shape: batch x query_size
        # values shape: batch x num values x value_size

        # attn_logits shape: batch x num values
        attn_logits = self.pointer(query, values, attn_mask)
        # attn_logits shape: batch x num values
        attn = self.softmax(attn_logits)
        # output shape: batch x 1 x value_size
        output = torch.bmm(attn.unsqueeze(1), values)
        output = output.squeeze(1)
        return output, attn


@registry.register('pointer', 'sdp')
class ScaledDotProductPointer(torch.nn.Module):
    def __init__(self, query_size, key_size):
        super().__init__()
        self.query_proj = torch.nn.Linear(query_size, key_size)
        self.temp = np.power(key_size, 0.5)
    
    def forward(self, query, keys, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # proj_query shape: batch x key_size x 1
        proj_query = self.query_proj(query).unsqueeze(2)
        
        # attn_logits shape: batch x num keys
        attn_logits = torch.bmm(keys, proj_query).squeeze(2) / self.temp
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


@registry.register('attention', 'sdp')
class ScaledDotProductAttention(Attention):
    def __init__(self, query_size, value_size):
        super().__init__(ScaledDotProductPointer(query_size, value_size))


@registry.register('pointer', 'bahdanau')
class BahdanauPointer(torch.nn.Module):
    def __init__(self, query_size, key_size, proj_size):
        super().__init__()
        self.compute_scores = torch.nn.Sequential(
            torch.nn.Linear(query_size + key_size, proj_size),
            torch.nn.Tanh(),
            torch.nn.Linear(proj_size, 1))
    
    def forward(self, query: torch.Tensor, keys: torch.Tensor, attn_mask=None):
        # query shape: batch x query_size
        # keys shape: batch x num keys x key_size

        # query_expanded shape: batch x num keys x query_size
        query_expanded = query.unsqueeze(1).expand(-1, keys.shape[1], -1)

        # scores shape: batch x num keys x 1
        attn_logits = self.compute_scores(
            # shape: batch x num keys x query_size + key_size
            torch.cat((query_expanded, keys),
            dim=2))
        # scores shape: batch x num keys
        attn_logits = attn_logits.squeeze(2)
        maybe_mask(attn_logits, attn_mask)
        return attn_logits


@registry.register('attention', 'bahdanau')
class BahdanauAttention(Attention):
    def __init__(self, query_size, value_size, proj_size):
        super().__init__(BahdanauPointer(query_size, value_size, proj_size))


# Adapted from The Annotated Transformers
class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, query_size, value_size, dropout=0.1, range_attention=False):
        super().__init__()
        assert query_size % h == 0
        assert value_size % h == 0

        # We assume d_v always equals d_k
        self.d_k = value_size // h
        self.h = h

        self.linears = torch.nn.ModuleList([
            torch.nn.Linear(query_size, value_size),
            torch.nn.Linear(value_size, value_size),
            torch.nn.Linear(value_size, value_size),
            torch.nn.Linear(value_size, value_size),
        ])

        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)
        self.range_attention = range_attention
        
    def forward(self, query, values, attn_mask=None, sep_id=None):
        "Implements Figure 2"
        if attn_mask is not None:
            # Same mask applied to all h heads.
            attn_mask = attn_mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, keys, values = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, values, values))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        # x, self.attn = transformer.sparse_attention(
        if not self.range_attention:
            x, self.attn = transformer.attention(
                    query, keys, values, mask=attn_mask, dropout=self.dropout)
        else:
            x, self.attn = range_attention(
                query, keys, values, sep_id=sep_id, mask=attn_mask, dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        x = x.squeeze(1)
        return self.linears[-1](x), self.attn


def range_attention(query, key, value, sep_id=None, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if sep_id is not \
            None:
        span_positions = []
        for i in range(len(sep_id) - 1):
            span_positions.append((sep_id[i], sep_id[i + 1]))
        span_positions.append((sep_id[-1], query.size(-1)))
        k_sep = key[:, :, sep_id]
        q_sep = k_sep[:, :, -1:]
        sep_attn_score = torch.matmul(q_sep, k_sep.transpose(-2, -1)) / math.sqrt(d_k)
        sep_attn = F.softmax(sep_attn_score, dim=-1)
        for i in range(sep_attn.size(-1)):
            scores[:, :, :, span_positions[i][0]:span_positions[i][1]] = \
                scores[:, :, :, span_positions[i][0]:span_positions[i][1]] + sep_attn[:, :, :, i].unsqueeze(-1)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # return torch.matmul(p_attn, value), scores.squeeze(1).squeeze(1)
    return torch.matmul(p_attn, value), p_attn
