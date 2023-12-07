# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from collections import OrderedDict
from typing import List, Optional
from lcasr.components.subsampling import StackingSubsampling

from abc import ABC, abstractmethod
from typing import List
import sentencepiece
import os


import torch
import torch.nn as nn
from einops import rearrange
import numpy as np
from typing import Union, Dict

from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from lcasr.components.dynamicpos import DynamicPositionBias
from lcasr.components.convolution import ConformerConvolution
from torch.nn import functional as F
from torch import einsum 

class preprocessor_config(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        #register all kwargs as buffers
        for k, v in kwargs.items():
            self.register_buffer(k, torch.tensor(v))

class dummy_positional_encoding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        '''
        Will spawn an instance of the stored function/class if fn exists
        '''
        return x, None if not hasattr(self, 'fn') else self.fn

    def store_fn(self, fn):
        '''
        Store a function/class
        '''
        self.fn = fn

    def extend_pe(self, *args, **kwargs):
        return None


class TokenizerSpec(ABC):
    """
    Inherit this class to implement a new tokenizer.
    """

    @abstractmethod
    def text_to_tokens(self, text):
        pass

    @abstractmethod
    def tokens_to_text(self, tokens):
        pass

    @abstractmethod
    def tokens_to_ids(self, tokens):
        pass

    @abstractmethod
    def ids_to_tokens(self, ids):
        pass

    @abstractmethod
    def text_to_ids(self, text):
        pass

    @abstractmethod
    def ids_to_text(self, ids):
        pass

    def add_special_tokens(self, special_tokens: List[str]):
        raise NotImplementedError("To be implemented")

    @property
    def name(self):
        return type(self).__name__

class SentencePieceTokenizer(TokenizerSpec):
    '''
    Sentencepiecetokenizer https://github.com/google/sentencepiece.
        Args:
        model_path: path to sentence piece tokenizer model. To create the model use create_spt_model()
        special_tokens: either list of special tokens or dictionary of token name to token value
        legacy: when set to True, the previous behavior of the SentecePiece wrapper will be restored, 
            including the possibility to add special tokens inside wrapper.
    '''

    def __init__(
        self, model_path: str, special_tokens: Optional[Union[Dict[str, str], List[str]]] = None, legacy: bool = False
    ):
        if not model_path or not os.path.exists(model_path):
            raise ValueError(f"model_path: {model_path} is invalid")
        self.tokenizer = sentencepiece.SentencePieceProcessor()
        self.tokenizer.Load(model_path)

        self.original_vocab_size = self.tokenizer.get_piece_size()
        self.vocab_size = self.tokenizer.get_piece_size()
        self.legacy = legacy
        self.special_token_to_id = {}
        self.id_to_special_token = {}
        if special_tokens:
            if not self.legacy:
                raise ValueError(
                    "Special tokens must be None when legacy is set to False. Provide special tokens at train time."
                )
            self.add_special_tokens(special_tokens)

    def text_to_tokens(self, text):
        if self.legacy:
            tokens = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                tokens.extend(self.tokenizer.encode_as_pieces(text[idx:next_idx]))
                tokens.append(next_token)
                idx = next_idx + len(next_token)

            tokens.extend(self.tokenizer.encode_as_pieces(text[idx:]))
            return tokens

        return self.tokenizer.encode_as_pieces(text)

    def text_to_ids(self, text):
        if self.legacy:
            ids = []
            idx = 0
            last_idx = 0

            while 1:
                indices = {}

                for token in self.special_token_to_id:
                    try:
                        indices[token] = text[idx:].index(token)
                    except ValueError:
                        continue

                if len(indices) == 0:
                    break

                next_token = min(indices, key=indices.get)
                next_idx = idx + indices[next_token]

                ids.extend(self.tokenizer.encode_as_ids(text[idx:next_idx]))
                ids.append(self.special_token_to_id[next_token])
                idx = next_idx + len(next_token)

            ids.extend(self.tokenizer.encode_as_ids(text[idx:]))
            return ids

        return self.tokenizer.encode_as_ids(text)

    def tokens_to_text(self, tokens):
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        return self.tokenizer.decode_pieces(tokens)

    def ids_to_text(self, ids):
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()

        if self.legacy:
            text = ""
            last_i = 0

            for i, id in enumerate(ids):
                if id in self.id_to_special_token:
                    text += self.tokenizer.decode_ids(ids[last_i:i]) + " "
                    text += self.id_to_special_token[id] + " "
                    last_i = i + 1

            text += self.tokenizer.decode_ids(ids[last_i:])
            return text.strip()

        return self.tokenizer.decode_ids(ids)

    def token_to_id(self, token):
        if self.legacy and token in self.special_token_to_id:
            return self.special_token_to_id[token]

        return self.tokenizer.piece_to_id(token)

    def ids_to_tokens(self, ids):
        tokens = []
        for id in ids:
            if id >= self.original_vocab_size:
                tokens.append(self.id_to_special_token[id])
            else:
                tokens.append(self.tokenizer.id_to_piece(id))
        return tokens

    def tokens_to_ids(self, tokens):
        if isinstance(tokens, str):
            tokens = [tokens]
        ids = []
        for token in tokens:
            ids.append(self.token_to_id(token))
        return ids

    def add_special_tokens(self, special_tokens):
        if not self.legacy:
            raise AttributeError("Special Token addition does not work when legacy is set to False.")

        if isinstance(special_tokens, list):
            for token in special_tokens:
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1
        elif isinstance(special_tokens, dict):
            for token_name, token in special_tokens.items():
                setattr(self, token_name, token)
                if (
                    self.tokenizer.piece_to_id(token) == self.tokenizer.unk_id()
                    and token not in self.special_token_to_id
                ):
                    self.special_token_to_id[token] = self.vocab_size
                    self.id_to_special_token[self.vocab_size] = token
                    self.vocab_size += 1

    @property
    def pad_id(self):
        if self.legacy:
            pad_id = self.tokens_to_ids([self.pad_token])[0]
        else:
            pad_id = self.tokenizer.pad_id()
        return pad_id

    @property
    def bos_id(self):
        if self.legacy:
            bos_id = self.tokens_to_ids([self.bos_token])[0]
        else:
            bos_id = self.tokenizer.bos_id()
        return bos_id

    @property
    def eos_id(self):
        if self.legacy:
            eos_id = self.tokens_to_ids([self.eos_token])[0]
        else:
            eos_id = self.tokenizer.eos_id()
        return eos_id

    @property
    def sep_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.sep_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def cls_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.cls_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def mask_id(self):
        if self.legacy:
            return self.tokens_to_ids([self.mask_token])[0]
        else:
            raise NameError("Use function token_to_id to retrieve special tokens other than unk, pad, bos, and eos.")

    @property
    def unk_id(self):
        return self.tokenizer.unk_id()

    @property
    def additional_special_tokens_ids(self):
        """Returns a list of the additional special tokens (excluding bos, eos, pad, unk). Used to return sentinel tokens for e.g. T5."""
        special_tokens = set(
            [self.bos_token, self.eos_token, self.pad_token, self.mask_token, self.cls_token, self.sep_token]
        )
        return [v for k, v in self.special_token_to_id.items() if k not in special_tokens]

    @property
    def vocab(self):
        main_vocab = [self.tokenizer.id_to_piece(id) for id in range(self.tokenizer.get_piece_size())]
        special_tokens = [
            self.id_to_special_token[self.original_vocab_size + i]
            for i in range(self.vocab_size - self.original_vocab_size)
        ]
        return main_vocab + special_tokens

def load_tokenizer(model_path:str):
    tokenizer_spe = SentencePieceTokenizer(
        model_path=model_path,
    )
    return tokenizer_spe

def load_defaul_instance():
    return SelfConditionedConformerEncoder(
        vocab=128, feat_in=80, d_model=256, n_layers=12, n_heads=8
    )

def load_from_old_state_dict(path:str, instance):
    '''
    Load a state dict from a previous version of the model
    '''
    state_dict = torch.load(path, map_location='cpu')['model_state_dict']
    # remove the prefix of the state dict
    state_dict = OrderedDict([(k.replace('encoder.', ''), v) for k, v in state_dict.items()])
    del state_dict['preprocessor.featurizer.window']
    del state_dict['preprocessor.featurizer.fb']
    instance.load_state_dict(state_dict)
    return instance

class SelfConditionedConformerEncoder(nn.Module):
    """
    The encoder for ASR model of Self Confitioned Conformer.
    Based on this paper:
    'Conformer: Convolution-augmented Transformer for Speech Recognition' by Anmol Gulati et al.
    https://arxiv.org/abs/2005.08100

    Args:
        feat_in (int): the size of feature channels
        n_layers (int): number of layers of ConformerBlock
        d_model (int): the hidden size of the model
        feat_out (int): the size of the output features
            Defaults to -1 (means feat_out is d_model)
        subsampling (str): the method of subsampling, choices=['vggnet', 'striding']
            Defaults to striding.
        subsampling_factor (int): the subsampling factor which should be power of 2
            Defaults to 4.
        subsampling_conv_channels (int): the size of the convolutions in the subsampling module
            Defaults to -1 which would set it to d_model.
        ff_expansion_factor (int): the expansion factor in feed forward layers
            Defaults to 4.
        self_attention_model (str): type of the attention layer and positional encoding
            'rel_pos': relative positional embedding and Transformer-XL
            'abs_pos': absolute positional embedding and Transformer
            default is rel_pos.
        pos_emb_max_len (int): the maximum length of positional embeddings
            Defaulst to 5000
        n_heads (int): number of heads in multi-headed attention layers
            Defaults to 4.
        xscaling (bool): enables scaling the inputs to the multi-headed attention layers by sqrt(d_model)
            Defaults to True.
        untie_biases (bool): whether to not share (untie) the bias weights between layers of Transformer-XL
            Defaults to True.
        conv_kernel_size (int): the size of the convolutions in the convolutional modules
            Defaults to 31.
        conv_norm_type (str): the type of the normalization in the convolutional modules
            Defaults to 'batch_norm'.
        dropout (float): the dropout rate used in all layers except the attention layers
            Defaults to 0.1.
        dropout_emb (float): the dropout rate used for the positional embeddings
            Defaults to 0.1.
        dropout_att (float): the dropout rate used for the attention layer
            Defaults to 0.0.
    """

    def __init__(
        self,
        vocab,
        feat_in,
        n_layers,
        d_model,
        feat_out=-1,
        subsampling='striding',
        subsampling_factor=4,
        subsampling_conv_channels=-1,
        ff_expansion_factor=4,
        n_heads=4,
        att_context_size=None,
        self_condition=True,
        conv_kernel_size=31,
        conv_norm_type='batch_renorm',
        dropout=0.1,
        dropout_att=0.0,
        checkpoint_every_n_layers=0,
        shared_kv = True,
        temperature = 15.5, # only imped with cosine sim attn
    ):
        super().__init__()

        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        #self.preprocesor = preprocessor_config(


        d_ff = d_model * ff_expansion_factor
        self.d_model = d_model
        self._feat_in = feat_in
        self.scale = math.sqrt(self.d_model)
        if att_context_size:
            self.att_context_size = att_context_size
        else:
            self.att_context_size = [-1, -1]

        self.self_condition = self_condition

        if subsampling_conv_channels == -1:
            subsampling_conv_channels = d_model
        if subsampling and subsampling_factor > 1:
            if subsampling == 'stacking':
                self.pre_encode = StackingSubsampling(
                    subsampling_factor=subsampling_factor, feat_in=feat_in, feat_out=d_model
                )
            else:
                self.pre_encode = ConvSubsampling(
                    subsampling=subsampling,
                    subsampling_factor=subsampling_factor,
                    feat_in=feat_in,
                    feat_out=d_model,
                    conv_channels=subsampling_conv_channels,
                    activation=nn.ReLU(),
                )
        else:
            self.pre_encode = nn.Linear(feat_in, d_model)

        self._feat_out = d_model

        self.decoder = ConvASRSelfConditioningDecoder(
            feat_in=d_model,
            num_classes=vocab,
        )

        pos_bias_u = None
        pos_bias_v = None
        dynamicpos = DynamicPositionBias(
            dim = d_model // 4,
            heads = n_heads,
            depth = 2,
            log_distance= False,
            norm = False
        )
        self.pos_enc = dummy_positional_encoding()
        self.pos_enc.store_fn(dynamicpos)

    
        self.layers = nn.ModuleList()
        for i in range(n_layers):
            layer = ConformerLayer(
                layer_idx = i,
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                conv_norm_type=conv_norm_type,
                dropout=dropout,
                dropout_att=dropout_att,
                shared_kv = shared_kv,
                temperature = temperature,
            )
            self.layers.append(layer)

        if feat_out > 0 and feat_out != self._feat_out:
            self.out_proj = nn.Linear(self._feat_out, feat_out)
            self._feat_out = feat_out
        else:
            self.out_proj = None
            self._feat_out = d_model
        self.use_pad_mask = True

    def set_max_audio_length(self, max_audio_length):
        """
        Sets maximum input length.
        Pre-calculates internal seq_range mask.
        """
        self.max_audio_length = max_audio_length
        device = next(self.parameters()).device
        seq_range = torch.arange(0, self.max_audio_length, device=device)
        if hasattr(self, 'seq_range'):
            self.seq_range = seq_range
        else:
            self.register_buffer('seq_range', seq_range, persistent=False)
        self.pos_enc.extend_pe(max_audio_length, device)

    @staticmethod
    def create_custom_forward(module): # for activation checkpointing allow passing dictionary as the argument to the module
        def custom_forward(*args, **kwargs):
            return module(*args, **kwargs)
        return custom_forward

    def forward(self, audio_signal, length=None):
        return self.forward_for_export(audio_signal=audio_signal, decoder=self.decoder, length=length)

    def forward_for_export(self, audio_signal, decoder, length):
        max_audio_length: int = audio_signal.size(-1)


        if length is None:
            length = torch.tensor([max_audio_length]*audio_signal.size(0), device=audio_signal.device)

        audio_signal = torch.transpose(audio_signal, 1, 2) 

        if isinstance(self.pre_encode, nn.Linear):
            audio_signal = self.pre_encode(audio_signal)
        else:
            audio_signal, length = self.pre_encode(audio_signal, length)

        audio_signal, pos_emb = self.pos_enc(audio_signal)
        # adjust size
        max_audio_length = audio_signal.size(1)
        # Create the self-attention and padding masks

        pad_mask = torch.arange(max_audio_length, device=length.device).unsqueeze(0) >= length.unsqueeze(1)
        print(pad_mask.shape)
        att_mask = pad_mask.unsqueeze(1).repeat([1, max_audio_length, 1])
        att_mask = torch.logical_and(att_mask, att_mask.transpose(1, 2))
        if self.att_context_size[0] >= 0:
            att_mask = att_mask.triu(diagonal=-self.att_context_size[0])
        if self.att_context_size[1] >= 0:
            att_mask = att_mask.tril(diagonal=self.att_context_size[1])
        att_mask = ~att_mask

        if self.use_pad_mask:
            pad_mask = ~pad_mask
        else:
            pad_mask = None

        iterim_posteriors = []
        for lth, layer in enumerate(self.layers):
            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal = checkpoint(self.create_custom_forward(layer), audio_signal, att_mask, pos_emb, pad_mask)
            else:
                audio_signal = layer(x=audio_signal, att_mask=att_mask, pos_emb=pos_emb, pad_mask=pad_mask)
                
            if lth != len(self.layers) - 1:
                iterim_logits = decoder(encoder_output=audio_signal.transpose(1, 2), logits=True)
                iterim_post = torch.nn.functional.softmax(iterim_logits, dim=-1)
                iterim_logposteriors = torch.log(iterim_post)
                iterim_posteriors.append(iterim_logposteriors)
                if self.self_condition == True:
                    audio_signal = decoder.integrate_projections(audio_signal, decoder.project_back(iterim_post))
                

        if self.out_proj is not None:
            audio_signal = self.out_proj(audio_signal) # if dim of decoder is not equal to dim of encoder, then we need to project the output
     
        audio_signal = torch.transpose(audio_signal, 1, 2) # (batch, seq_len, d_model) -> (batch, d_model, seq_len) 

        # stack the posteriors along the first dimension (height, batch, seq_len, dim)
        iterim_posteriors = torch.stack(iterim_posteriors, dim=0)
        
        final_posts = decoder(encoder_output=audio_signal, logits=False)

        #return audio_signal, iterim_posteriors, length

        return {
            'final_posteriors': final_posts,
            'iterim_posteriors': iterim_posteriors,
            'length': length,
        }

def Swish():
    return nn.SiLU()

class ConformerFeedForward(nn.Module):
    """
    feed-forward module of Conformer model.
    """

    def __init__(self, d_model, d_ff, dropout, activation=Swish()):
        super(ConformerFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = activation
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class ConformerLayer(torch.nn.Module):
    """A single block of the Conformer encoder.

    Args:
        d_model (int): input dimension of MultiheadAttentionMechanism and PositionwiseFeedForward
        d_ff (int): hidden dimension of PositionwiseFeedForward
        n_heads (int): number of heads for multi-head attention
        conv_kernel_size (int): kernel size for depthwise convolution in convolution module
        dropout (float): dropout probabilities for linear layers
        dropout_att (float): dropout probabilities for attention distributions
    """

    def __init__(
        self,
        layer_idx,
        d_model,
        d_ff,
        n_heads=4,
        conv_kernel_size=31,
        conv_norm_type='batch_renorm',
        dropout=0.1,
        dropout_att=0.1,
        talking_heads = 'none', # only implemented with cosine attn atm 'pre' or 'post' or 'both' or 'none'
        shared_kv = True,
        temperature = 15.5,
    ):
        super(ConformerLayer, self).__init__()


        self.n_heads = n_heads
        self.fc_factor = 0.5

        self.layer_idx = layer_idx


        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        ff1_d_ff = d_ff
        ff1_d0 = dropout


        self.feed_forward1 = ConformerFeedForward(d_model=d_model, d_ff=ff1_d_ff, dropout=ff1_d0)

        # convolution module
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConformerConvolution(
            d_model=d_model, 
            kernel_size=conv_kernel_size, 
            norm_type=conv_norm_type, 
        )


        # multi-headed self-attention module
        self.norm_self_att = nn.LayerNorm(d_model)
 
 
        self.self_attn = CosineAttention(
            n_feats = d_model,
            head_dim = max(16, d_model // n_heads),
            n_heads = n_heads,
            bias = False,
            temperature = temperature,
            causal = False,
            shared_kv = shared_kv,
            talking_heads = talking_heads,
            dropout = dropout_att,
            spatial_attention_dropout = False
        )



        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = ConformerFeedForward(d_model=d_model, d_ff=d_ff, dropout=dropout)


        self.dropout = nn.Dropout(dropout)
        self.norm_out = nn.LayerNorm(d_model)

    @staticmethod
    def slice_mem_tokens(x, num_mem_tokens):
        return x[:, :num_mem_tokens, :], x[:, num_mem_tokens:, :]

    def forward(self, x, att_mask=None, pos_emb=None, pad_mask=None, num_memory_vectors=None, mem_pos_emb=None, return_attentions=False):
        """
        Args:
            x (torch.Tensor): input signals (B, T, d_model)
            att_mask (torch.Tensor): attention masks(B, T, T)
            pos_emb (torch.Tensor): (L, 1, d_model)
            pad_mask (torch.tensor): padding mask
        Returns:
            x (torch.Tensor): (B, T, d_model)
        """
        residual = x
        x = self.norm_feed_forward1(x)
        x = self.feed_forward1(x)
        residual = residual + self.dropout(x) * self.fc_factor


        x = self.norm_self_att(residual)
   
   
        x = self.self_attn(x=x, pos_fn=pos_emb, mask=pad_mask)
    
  
        x, attns = x if return_attentions else (x, None)
        residual = residual + self.dropout(x)

        conv_pad_mask = pad_mask


        x = self.norm_conv(residual)
        x = self.conv(x, conv_pad_mask)

        residual = residual + self.dropout(x)

     
        x = self.norm_feed_forward2(residual)
        x = self.feed_forward2(x)
        residual = residual + self.dropout(x) * self.fc_factor
        x = self.norm_out(residual) 


        return x if not return_attentions else (x, attns)

class ConvASRSelfConditioningDecoder(nn.Module):
    """Simple ASR Decoder for use with CTC-based models such as JasperNet and QuartzNet

     Based on these papers:
        https://arxiv.org/pdf/1904.03288.pdf
        https://arxiv.org/pdf/1910.10261.pdf
        https://arxiv.org/pdf/2005.04290.pdf
    """


    def __init__(
        self, 
        feat_in, 
        num_classes, 
        init_mode="xavier_uniform", 
        vocabulary=None, 
        remove_ctx=False, # legacy
        gating=False, # gating didn't help rlly
        reproject_type='linear', # linear, conv (using linear), concat
        auxilary_training=False # train reprojection layer using word embedding task (NOT IMPLEMENTED)
    ):
        super().__init__()

        assert not (reproject_type == 'concat' and gating), "concat_reprojection and gating can't be used together"
        assert auxilary_training == False, "Not implemented yet"
        assert reproject_type in ['linear', 'conv', 'concat', 'none'], "reproject_type should be either 'linear' or 'conv' or 'concat' or none (inter-ctc)"
        if auxilary_training == True:
            assert reproject_type == 'linear', "reproject_type should be 'linear' when auxilary_training is True"
        
        self.reproject_type = reproject_type

        self.auxilary_training = auxilary_training

        if self.auxilary_training == True: # Legacy
            self.left_word_prediction = torch.nn.Linear(
                feat_in,
                num_classes
            )
            self.right_word_prediction = torch.nn.Linear(
                feat_in,
                num_classes
            )

        if vocabulary is None and num_classes < 0:
            raise ValueError(
                f"Neither of the vocabulary and num_classes are set! At least one of them need to be set."
            )

        if num_classes <= 0:
            num_classes = len(vocabulary)
            

        if vocabulary is not None:
            if num_classes != len(vocabulary):
                raise ValueError(
                    f"If vocabulary is specified, it's length should be equal to the num_classes. Instead got: num_classes={num_classes} and len(vocabulary)={len(vocabulary)}"
                )
            self.__vocabulary = vocabulary
        self._feat_in = feat_in
        # Add 1 for blank char
        self._num_classes = num_classes + 1

        self.decoder_layers = torch.nn.Sequential(
            torch.nn.Conv1d(self._feat_in, self._num_classes, kernel_size=1, bias=True)
        )

        if reproject_type != 'none':
            self.reprojection_layers = torch.nn.Sequential( # project from logspace back to model dim. Change to linear !
                torch.nn.Conv1d(self._num_classes, self._feat_in, kernel_size=1, bias=True) if reproject_type == 'conv' else torch.nn.Linear(self._num_classes, self._feat_in, bias=True)
            ) if reproject_type != 'concat' else torch.nn.Linear(self._num_classes + self._feat_in, self._feat_in, bias=True)


        self.gating = gating
        if self.gating:   
            self.weight_hidden = torch.nn.Linear(self._feat_in, self._feat_in, bias=False)
            self.weight_embedding = torch.nn.Linear(self._feat_in, self._feat_in, bias=False)
            self.gating_bias = torch.nn.Parameter(torch.zeros(self._feat_in))
            self.gating_sigmoid = torch.nn.Sigmoid()
        
        self.remove_ctx = remove_ctx
      

    def forward(self, encoder_output, logits=False):

        
        if self.remove_ctx == False:
            out = self.decoder_layers(encoder_output).transpose(1, 2)
        else: # remove first iterm in each sequence 
            out = self.decoder_layers(encoder_output[:,:,1:]).transpose(1, 2)
        
        if logits == False: 
            out = torch.nn.functional.log_softmax(out, dim=-1)

        return out

    def project_back(self, decoder_output):
        '''
        Projects the decoder output back to the acoustic models hidden dimension for self-conditioning.
        '''
        if self.reproject_type != 'concat':
            return self.reprojection_layers(decoder_output.transpose(1, 2)).transpose(1, 2) if self.reproject_type == 'conv' else self.reprojection_layers(decoder_output)
        else:
            return decoder_output # do nothing atm

    def integrate_projections(self, encoder_out, proj1):
        if self.gating == False:
            if self.reproject_type == 'concat':
                concat_seq = torch.cat((encoder_out, proj1), dim=-1) # concat along the feature dimension
                return self.reprojection_layers(concat_seq) # project back to the encoder dimension
            else:
                return encoder_out + proj1
        else:
            embedding = proj1
            gating_fn = self.gating_sigmoid(self.weight_hidden(encoder_out) + self.weight_embedding(embedding) + self.gating_bias)
            encoder_new = encoder_out * gating_fn + embedding * (1 - gating_fn)
            return encoder_new

    def auxilary_task(self, input_tokens): # NOT DONE
        assert self.auxilary_training == True, "auxilary_training is not set to True"
        word_embeddings = self.reprojection_layers.weight.T.squeeze()[input_tokens]
        left_word_prediction = self.left_word_prediction(word_embeddings)
        right_word_prediction = self.right_word_prediction(word_embeddings)
        
        return left_word_prediction, right_word_prediction

    def input_example(self, max_batch=1, max_dim=256):
        """
        Generates input examples for tracing etc.
        Returns:
            A tuple of input examples.
        """
        input_example = torch.randn(max_batch, self._feat_in, max_dim).to(next(self.parameters()).device)
        return tuple([input_example])

    @property
    def vocabulary(self):
        return self.__vocabulary

    @property
    def num_classes_with_blank(self):
        return self._num_classes

def l2norm(t, groups = 1, dim = -1):
    if groups == 1:
        return F.normalize(t, p = 2, dim = dim)
    t = rearrange(t, '... (g d) -> ... g d', g = groups)
    t = F.normalize(t, p = 2, dim = dim)
    return rearrange(t, '... g d -> ... (g d)')

class CosineAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.1,
        bias=False,
        temperature=15.5,
        return_attention=False,
        causal=False,
        **kwargs
    ):
        super().__init__()
        
        self.shared_kv = kwargs.get('shared_kv', False)
        self.talking_heads = kwargs.get('talking_heads', 'pre')
        

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
        
        self.dropout = nn.Dropout(dropout)
      
        self.bias = bias
        self.return_attention = return_attention
        self.causal = causal

        if self.talking_heads:
            if self.talking_heads == 'pre' or self.talking_heads == 'both':
                self._head_proj = nn.Conv2d(n_heads, n_heads, (1, 1))
            if self.talking_heads == 'post' or self.talking_heads == 'both':
                self._head_proj_post = nn.Conv2d(n_heads, n_heads, (1, 1))
            

        self.temperature = torch.nn.Parameter(torch.tensor(temperature), requires_grad=True) if isinstance(temperature, float) else temperature

        self.activation = nn.Softmax(dim=-1)

        if not self.shared_kv:
            self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
            self.qkv = lambda x: rearrange(self.qkv_proj(x), "b n (h d qkv) -> qkv b h n d", qkv=3, h=n_heads, d=head_dim)
        else:
            self.q_proj, self.kv_proj = [nn.Linear(n_feats, el, bias=bias) for el in [n_heads * head_dim, 2 * head_dim]]
            map_q, map_kv = lambda q: rearrange(q, 'b n (h d) -> b h n d', h=n_heads), lambda kv: rearrange(kv, 'b n (kv d) -> kv b () n d', kv=2, d=head_dim)
            self.qkv = lambda x: (map_q(self.q_proj(x)), *map_kv(self.kv_proj(x)))

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)
    
    def head_proj(self, dots, mode='pre'):
        if mode == 'pre' and (self.talking_heads == 'pre' or self.talking_heads == 'both'):
            dots = self._head_proj(dots)
        if mode == 'post' and (self.talking_heads == 'post' or self.talking_heads == 'both'):
            dots = self._head_proj_post(dots)
        return dots      

    def attend(self, query, key, value, mask, pos_fn):
        query, key = map(l2norm, (query, key))

        dots = einsum('bhid,bhjd->bhij', query, key) * self.temperature
        dots = self.head_proj(dots, mode='pre')

        dots += pos_fn(dots.shape[-1], device=dots.device, dtype=dots.dtype)
        qkmask = ~mask
        attn_mask = ~(rearrange(qkmask, "b n -> b () n ()") * rearrange(qkmask, "b n -> b () () n"))
    
        if self.causal: # create a regular causal mask
            causal_mask = torch.ones(dots.shape[-2], dots.shape[-1], device=dots.device).triu(1).bool()
            attn_mask = torch.logical_or(attn_mask, causal_mask)
        
        dots.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)
    
        attn = self.activation(dots)
        attn = self.head_proj(attn, mode='post')
    
        attn = self.dropout(attn)
        return einsum("bhij,bhjd->bhid", attn, value)


    def forward(self, x, pos_fn, mask=None):
        assert pos_fn is not None, 'pls provide a position function'
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim
        #print(x.shape, mask.shape)
       
        if mask is None:
            mask = torch.zeros(B, N, device=x.device, dtype=torch.bool)

        q, k, v = self.qkv(x)
    
        out = self.attend(q, k, v, mask, pos_fn)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out


class ConvSubsampling(torch.nn.Module):
    """Convolutional subsampling which supports VGGNet and striding approach introduced in:
    VGGNet Subsampling: Transformer-transducer: end-to-end speech recognition with self-attention (https://arxiv.org/pdf/1910.12977.pdf)
    Striding Subsampling: "Speech-Transformer: A No-Recurrence Sequence-to-Sequence Model for Speech Recognition" by Linhao Dong et al. (https://ieeexplore.ieee.org/document/8462506)
    Args:
        subsampling (str): The subsampling technique from {"vggnet", "striding"}
        subsampling_factor (int): The subsampling factor which should be a power of 2
        feat_in (int): size of the input features
        feat_out (int): size of the output features
        conv_channels (int): Number of channels for the convolution layers.
        activation (Module): activation function, default is nn.ReLU()
    """

    def __init__(self, subsampling, subsampling_factor, feat_in, feat_out, conv_channels, activation=nn.ReLU(), stride=2, kernel_size=3):
        super(ConvSubsampling, self).__init__()
        self._subsampling = subsampling

        if subsampling_factor % 2 != 0:
            raise ValueError("Sampling factor should be a multiply of 2!")
        self._sampling_num = int(math.log(subsampling_factor, 2))

        in_channels = 1
        layers = []
        self._ceil_mode = False

        if subsampling == 'vggnet':
            self._padding = 0
            self._stride = stride
            self._kernel_size = kernel_size
            self._ceil_mode = True

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=conv_channels, out_channels=conv_channels, kernel_size=3, stride=1, padding=1
                    )
                )
                layers.append(activation)
                layers.append(
                    torch.nn.MaxPool2d(
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                        ceil_mode=self._ceil_mode,
                    )
                )
                in_channels = conv_channels
        elif subsampling == 'striding':
            self._padding = 1
            self._stride = stride
            self._kernel_size = kernel_size
            self._ceil_mode = False

            for i in range(self._sampling_num):
                layers.append(
                    torch.nn.Conv2d(
                        in_channels=in_channels,
                        out_channels=conv_channels,
                        kernel_size=self._kernel_size,
                        stride=self._stride,
                        padding=self._padding,
                    )
                )
                layers.append(activation)
                in_channels = conv_channels
        else:
            raise ValueError(f"Not valid sub-sampling: {subsampling}!")

        in_length = torch.tensor(feat_in, dtype=torch.float)
        out_length = calc_length(
            in_length,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        self.out = torch.nn.Linear(conv_channels * int(out_length), feat_out)
        self.conv = torch.nn.Sequential(*layers)

    def forward(self, x, lengths):
        lengths = calc_length(
            lengths,
            padding=self._padding,
            kernel_size=self._kernel_size,
            stride=self._stride,
            ceil_mode=self._ceil_mode,
            repeat_num=self._sampling_num,
        )
        x = x.unsqueeze(1)
        if self._subsampling == 'striding':
            # added in order to prevent slowdown in torch.nn.Conv2d with bfloat16 / CUDNN v8 API
            # to be removed once the above is fixed in cudnn
            with torch.cuda.amp.autocast(dtype=torch.float32):
                x = self.conv(x)
        else:
            x = self.conv(x)

        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).reshape(b, t, -1))
        return x, lengths


def calc_length(lengths, padding, kernel_size, stride, ceil_mode, repeat_num=1):
    """ Calculates the output length of a Tensor passed through a convolution or max pooling layer"""
    add_pad: float = (padding * 2) - kernel_size
    one: float = 1.0
    for i in range(repeat_num):
        lengths = torch.div(lengths.to(dtype=torch.float) + add_pad, stride) + one
        if ceil_mode:
            lengths = torch.ceil(lengths)
        else:
            lengths = torch.floor(lengths)
    return lengths.to(dtype=torch.int)