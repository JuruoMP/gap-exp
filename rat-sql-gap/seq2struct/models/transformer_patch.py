import random
import logging

import torch
import torch.nn as nn
from transformers import BartConfig, BartModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.bart.modeling_bart import BartEncoder, _expand_mask


class BartMultiSegmentModel(BartModel):
    class BartMultiSegmentEncoder(BartEncoder):
        def __init__(self, bart_encoder, max_segment):
            super().__init__(bart_encoder.config)
            self.dropout = bart_encoder.dropout
            self.layerdrop = bart_encoder.layerdrop
            self.padding_idx = bart_encoder.padding_idx
            self.max_source_positions = bart_encoder.max_source_positions
            self.embed_scale = bart_encoder.embed_scale
            self.embed_tokens = bart_encoder.embed_tokens
            self.embed_positions = bart_encoder.embed_positions
            self.layers = bart_encoder.layers
            self.layernorm_embedding = bart_encoder.layernorm_embedding
            # additional segment embedding
            self.segment_embedding = nn.Embedding(max_segment, self.config.d_model)

        def forward(
                self,
                input_ids=None,
                segment_ids=None,
                attention_mask=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
        ):
            r"""
            Args:
                input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
                    Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                    provide it.

                    Indices can be obtained using :class:`~transformers.BartTokenizer`. See
                    :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__`
                    for details.

                    `What are input IDs? <../glossary.html#input-ids>`__
                attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
                    Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

                    - 1 for tokens that are **not masked**,
                    - 0 for tokens that are **masked**.

                    `What are attention masks? <../glossary.html#attention-mask>`__
                head_mask (:obj:`torch.Tensor` of shape :obj:`(encoder_layers, encoder_attention_heads)`, `optional`):
                    Mask to nullify selected heads of the attention modules. Mask values selected in ``[0, 1]``:

                    - 1 indicates the head is **not masked**,
                    - 0 indicates the head is **masked**.

                inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
                    Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded
                    representation. This is useful if you want more control over how to convert :obj:`input_ids` indices
                    into associated vectors than the model's internal embedding lookup matrix.
                output_attentions (:obj:`bool`, `optional`):
                    Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under
                    returned tensors for more detail.
                output_hidden_states (:obj:`bool`, `optional`):
                    Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors
                    for more detail.
                return_dict (:obj:`bool`, `optional`):
                    Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
            """
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # retrieve input_ids and inputs_embeds
            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

            embed_pos = self.embed_positions(input_shape)

            hidden_states = inputs_embeds + embed_pos
            if segment_ids is not None:
                embed_segment = self.segment_embedding(segment_ids)
                hidden_states += embed_segment
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            # check if head_mask has a correct number of layers specified if desired
            if head_mask is not None:
                assert head_mask.size()[0] == (
                    len(self.layers)
                ), f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if getattr(self.config, "gradient_checkpointing", False) and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            )

    def __init__(self, config):
        super().__init__(config)
        self.patched = False

    def patch_segment(self, max_segment):
        if self.patched:
            return
        bart_multi_segment_encoder = BartMultiSegmentModel.BartMultiSegmentEncoder(self.encoder, max_segment)
        self.encoder = bart_multi_segment_encoder
        self.patched = True
