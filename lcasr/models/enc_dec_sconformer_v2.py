import torch, torch.nn as nn, torch.nn.functional as F
import random
from torch.utils.checkpoint import checkpoint # # gradient/activation checkpointing
from einops import rearrange, repeat
from typing import Dict, List, Tuple, Union
from functools import partial
from lcasr.components import fused_dense, subsampling, convolution, decoder, wrappers
from lcasr.components.positional_encodings import RotaryPositionalEmbedding, apply_rotary, LearnableFourierPosEnc, DynamicPositionBias
from lcasr.utils.helpers import exists
from lcasr.components.wrappers import Scale
from lcasr.utils.lm_tools import add_eos, token_lens_to_mask, mark_padding
ConformerConvolution = convolution.ConformerConvolution
ConformerFeedForward = fused_dense.FusedMLP
ConvSubsampling, StackingSubsampling = subsampling.ConvSubsampling, subsampling.StackingSubsampling
ConvResidualSubsampling = subsampling.ConvResidualSubsampling
try: from apex.normalization import FusedRMSNorm as DEFAULT_NORM, FusedRMSNorm as RMSNorm, FusedLayerNorm as LayerNorm
except: 
    from lcasr.components.normalisation import RMSNorm as RMSNorm, RMSNorm as DEFAULT_NORM
    from torch.nn import LayerNorm as LayerNorm
PreNorm, Scale = wrappers.PreNorm, wrappers.Scale


from lcasr.components.attention import Attention
try:
    from flash_attn.modules.mha import FlashCrossAttention
    from flash_attn.bert_padding import unpad_input, pad_input
except ImportError:
    FlashCrossAttention = None
    unpad_input = pad_input = None


from lcasr.components.helpers import get_act
from lcasr.models.base import BaseModel
from torch import einsum
import math
import warnings
from lcasr.decoding import ctc_beam_search
import sentencepiece as spm



class EncDecSconformerV2(BaseModel): 
    def __init__(
        self,
        vocab_size = 4096,
        feat_in = 80,
        subsampling = 'dw_striding',
        subsampling_factor = 8,
        subsampling_conv_channels = 256,
        subsampling_act = 'silu',
        subsampling_norm_out = False,
        self_condition_subsampling = False,
        n_layers = 3,
        d_model = 768,
        n_heads = 6,
        head_dim = 128,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_conv = 0.0,
        dropout_attn = 0.0,
        checkpoint_every_n_layers = 0,
        conv_kernel_size = 9,
        conv_expansion_factor = 1,
        decoder_norm = True,
        use_rotary = True,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        learned_rotary = False,
        self_conditioning = True,
        default_norm = 'layer_norm',
        sandwich_norm = False,
        bias_in_ff = False,
        transformer=False, # disable convolutions
        ctc_loss_weight = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.feat_in = feat_in
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.expansion_factor = expansion_factor
        self.conv_kernel_size = conv_kernel_size
        self.conv_expansion_factor = conv_expansion_factor
        self.rotary_interpolation_factor = rotary_interpolation_factor
        self.learned_rotary = learned_rotary
        self.self_conditioning = self_conditioning if ctc_loss_weight > 0 else False # no ctc decoder
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.transformer = transformer
        self.self_condition_subsampling = self_condition_subsampling

        self.ctc_loss_weight = ctc_loss_weight

        # self.abs_pos_enc = PosEnc(d_model)
        self.use_abs_pos_enc = kwargs.get('use_abs_pos_enc', True)
        self.pos_enc = LearnableFourierPosEnc(d_model, hidden_dim=kwargs.get('fourier_pos_hidden_dim', 64)) if self.use_abs_pos_enc else nn.Identity()

        self.checkpoint_subsampling = kwargs.get('checkpoint_subsampling', False) # whether to perform activation checkpointing on subsampling layers

        accepted_norms = ['rms_norm', 'layer_norm']
        accepted_subsampling_acts = ['silu', 'relu', 'gelu', 'none']
        assert subsampling_act in accepted_subsampling_acts, f'subsampling_act must be one of {accepted_subsampling_acts} (got {subsampling_act})'
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else LayerNorm

        subsampling_act = get_act(subsampling_act)

        self.flash_attn = kwargs.get('flash_attn', True)
        self.checkpoint_every_n_layers = checkpoint_every_n_layers

        self.dropout_ff = dropout_ff
        self.dropout_conv = dropout_conv
        self.dropout_attn = dropout_attn

        self.subsampling_mode = subsampling
        self.subsampling_factor = subsampling_factor
        self.subsampling_conv_channels = subsampling_conv_channels if subsampling_conv_channels != -1 else d_model

        self.decoder_norm = decoder_norm

        self.use_rotary = use_rotary

        self.rotary_pos_emb = None
        if self.use_rotary:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim = head_dim,
                base = kwargs.get('rotary_base_freq', 10000),
                learned_freq = learned_rotary,
                rotary_interpolation_factor = rotary_interpolation_factor
            )

        self.ctc_decoder = decoder.ASRLinearSCDecoder(
            d_model = d_model,
            vocab_size = vocab_size,
            norm = decoder_norm,
            norm_fn = default_norm,
        ) if ctc_loss_weight > 0 else None

        self.language_model_decoder = CrossAttnDecoder(
            vocab_size = vocab_size,
            n_layers = n_layers,
            d_model = d_model,
            n_heads = n_heads,
            head_dim = head_dim,
            expansion_factor = expansion_factor,
            dropout_attn = dropout_attn,
            dropout_ff = dropout_ff,
            decoder_norm=decoder_norm,
            bias_in_ff=bias_in_ff,
            **kwargs
        )

        subsampling_args = {'subsampling_factor': self.subsampling_factor, 'feat_in': self.feat_in, 'feat_out': self.d_model, 'norm_out': subsampling_norm_out,}
        if subsampling == 'stacking':
            self.subsampling = StackingSubsampling(norm = True if not subsampling_norm_out else False, default_norm = default_norm, **subsampling_args)
        elif subsampling == 'conv_residual':
            self.subsampling = ConvResidualSubsampling(subsampling = self.subsampling_mode, conv_channels = self.subsampling_conv_channels, activation = subsampling_act, **subsampling_args) 
        else:
            self.subsampling = ConvSubsampling(subsampling = self.subsampling_mode, conv_channels = self.subsampling_conv_channels, activation = subsampling_act, **subsampling_args) 
     

        
        self.layers = nn.ModuleList()


        for i in range(n_layers):
            l = ConformerLayer(
                d_model = d_model,
                conv_kernel_size = conv_kernel_size,
                expansion_factor = expansion_factor,
                dropout_ff = dropout_ff,
                dropout_conv = dropout_conv,
                dropout_attn = dropout_attn,
                layer_idx = i,
                total_layers = n_layers,
                head_dim = head_dim,
                n_heads = n_heads,
                default_norm = default_norm,
                sandwich_norm = sandwich_norm,
                bias_in_ff = bias_in_ff,
                transformer = transformer,
                conv_expansion_factor = conv_expansion_factor,
                **kwargs
            )
            self.layers.append(l)

            self.no_encoder_padding = kwargs.get('no_encoder_padding', False)
            self.min_seq_len = kwargs.get('min_seq_len', 0)


    def calc_loss(
            self, 
            audio_signal, # B, C, T
            text_sequence,
            a_lengths,
            t_lengths,
            lm_text_sequence=None,
            lm_text_sequence_lengths=None,
            lm_loss_mask=None,
            bos_id=0, 
            eos_id=0,
            encoder_outputs=None,
            **kwargs
        ):

        other_outputs = {}
        if lm_text_sequence is None: # add bos to text sequence
            text_sequence_bos = F.pad(text_sequence, (1, 0), value=bos_id)
            target_lengths_bos = t_lengths + 1
        else:
            assert lm_text_sequence_lengths is not None, 'lm_text_sequence_lengths must be provided if lm_text_sequence is provided'
            text_sequence_bos = lm_text_sequence
            target_lengths_bos = lm_text_sequence_lengths
        
        if encoder_outputs is None: # run the encoder and decoder if hidden states are not provided
            out = self.forward(audio_signal, text_sequence_bos, a_lengths)
            ctc_out, lm_out, a_length_out = out['final_posteriors_ctc'], out['final_posteriors_lm'], out['length']
        else:
            ctc_out = encoder_outputs['final_posteriors_ctc']
            a_length_out = encoder_outputs['length']
            lm_out = self.language_model_decoder(
                tokens = text_sequence_bos,
                a_hidden = encoder_outputs['a_hidden'],
                a_lengths = a_lengths,
                cache = None,
            )
            lm_out = lm_out['logits']


        if self.ctc_loss_weight > 0.0:
            if kwargs.get('trim_ctc_by', None) is not None:
                trim_ctc_by = kwargs.get('trim_ctc_by', None)
                a_lengths = a_lengths - trim_ctc_by
                in_length = audio_signal.shape[-1]
                out_length = ctc_out.shape[1]
                downsample_factor = in_length / out_length
                trim_ctc_by = int(round(trim_ctc_by / downsample_factor))
                a_length_out = a_length_out - trim_ctc_by
                
            else: trim_ctc_by = 0

            ctc_loss = F.ctc_loss(
                log_probs = rearrange(ctc_out[:, trim_ctc_by:], 'b n c -> n b c'),
                targets = text_sequence,
                input_lengths = a_length_out,
                target_lengths = t_lengths,
                reduction = 'sum',
                blank = ctc_out.shape[-1] - 1
            )
            a_sum = a_lengths.sum()
            ctc_loss_to_show = (ctc_loss / a_sum).item() * 100
            ctc_loss_to_bwd = ctc_loss / (ctc_out[:, trim_ctc_by:].shape[1] * ctc_out.shape[0]) * 100
        else:
            ctc_loss_to_show, ctc_loss_to_bwd = 0, 0

        targets = text_sequence_bos.clone()
        targets[:, :-1] = text_sequence_bos[:, 1:]
        if target_lengths_bos.max() == target_lengths_bos.min(): targets[:, -1] = 0
        else:
            targets = add_eos(targets, eos_id = eos_id, token_lens = target_lengths_bos)
        mask = token_lens_to_mask(target_lengths_bos)
        targets = mark_padding(targets, mask, pad_id = -100)
        
        lm_num_masked = 0
        if lm_loss_mask is not None:
            assert lm_loss_mask.shape == targets.shape, f'lm_loss_mask shape {lm_loss_mask.shape} does not match targets shape {targets.shape}'
            lm_loss_mask = lm_loss_mask.to(targets.device)
            targets = targets.masked_fill(lm_loss_mask, -100)
            lm_num_masked = lm_loss_mask.sum()

            
        predictions = lm_out
        lm_loss = F.cross_entropy(
            input = rearrange(predictions, 'b n c -> (b n) c'),
            target = rearrange(targets, 'b n -> (b n)'),
            ignore_index = -100,
            reduction = 'none'
        )
        if kwargs.get('return_token_losses', False):
            lm_loss = rearrange(lm_loss, '(b n) -> b n', b = predictions.shape[0])
            other_outputs['lm_loss'] = lm_loss
            other_outputs['lm_loss_mask'] = lm_loss_mask

        lm_loss = lm_loss.sum()
        lm_loss_to_show = (lm_loss / (target_lengths_bos.sum() - lm_num_masked)).item() 
        lm_loss_to_bwd = lm_loss / ((predictions.shape[0] * predictions.shape[1]) - lm_num_masked) 

        loss_to_show = ctc_loss_to_show * self.ctc_loss_weight + lm_loss_to_show * (1 - self.ctc_loss_weight)
        loss = ctc_loss_to_bwd * self.ctc_loss_weight + lm_loss_to_bwd 

        wandb_log_data = {
            'loss': loss_to_show,
            'ctc_loss': ctc_loss_to_show,
            'lm_loss': lm_loss_to_show,
        }

        return {
            'loss': loss,
            'display_losses': wandb_log_data,
            'ctc_posteriors': ctc_out,
            'lm_posteriors': lm_out,
            'length': a_length_out,
            **other_outputs
        }
        
    @torch.no_grad()
    def generate(
            self,
            audio_signal=None,
            max_generate='encoder_states',
            bos_id=0,
            eos_id=0,
            return_encoder_states=False,
            return_ctc_states=False,
            prompt:Union[List[int], List[List[int]], torch.LongTensor]=None,
            encoder_states:Dict[str, torch.Tensor]=None,
            remove_prompt=True,
            sample=False,
            temperature=1.0,
            num_rollouts:int=1,
            beam_width:int=1,
            length_penalty:float=0.0,
            eos_bias:float=0.0,
            repetition_penalty:float=0.0,
            no_repeat_ngram_size:int=0,
            return_beam_scores:bool=False,
        ) -> Dict[str, Union[torch.Tensor, List[List[int]], List[List[float]]]]:
        '''
        Batched, kv-cached generation. Supports both greedy and multinomial sampling,
        with per-row early-exit on EOS (finished rows are dropped from the active batch).

        audio_signal: (B, ...) — required if encoder_states not provided
        max_generate: max tokens per sequence; 'encoder_states' caps at the encoder length
        bos_id / eos_id: tokens for prompt seeding / early termination
        return_encoder_states / return_ctc_states: include encoder/ctc outputs
        prompt: None (use bos), List[int] (broadcast across rows), List[List[int]]
            (per-row, must share length), or LongTensor of shape (B*num_rollouts, P)
        encoder_states: dict {'a_hidden', 'length'} to skip the encoder forward
        remove_prompt: strip the prompt tokens from each returned sequence
        sample: True -> multinomial sampling; False -> argmax (temperature ignored)
        temperature: softmax temperature when sampling
        num_rollouts: replicate each input row this many times in parallel (e.g. RL rollouts)
        beam_width: 1 -> greedy/sample path; >1 -> autoregressive beam search
        length_penalty: normalize beam scores by generated length**length_penalty
        eos_bias: added to EOS log-prob during beam search; positive encourages shorter outputs
        repetition_penalty: subtract from log-probs for tokens already generated in a beam
        no_repeat_ngram_size: block tokens that would repeat an ngram of this size
        return_beam_scores: include selected beam scores in the output dict

        Returns a dict with:
            text_sequence: List[List[int]]   — length B*num_rollouts
            probs:         List[List[float]] — chosen-token probability per step, per row
            beam_scores (optional)           — selected beam score per row
            encoder_states (optional), ctc_states (optional)
        '''

        if encoder_states is None:
            encoder_out = self.forward(audio_signal=audio_signal)
            a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
        else:
            encoder_out = encoder_states
            a_hidden, length = encoder_states['a_hidden'], encoder_states['length']

        if num_rollouts > 1:
            a_hidden = a_hidden.repeat_interleave(num_rollouts, dim=0)
            length = length.repeat_interleave(num_rollouts, dim=0)

        B = a_hidden.shape[0]
        device = a_hidden.device

        if max_generate == 'encoder_states': max_generate = length.max().item()

        # Build (B, P) prompt tensor.
        if prompt is None:
            text_sequence = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        elif isinstance(prompt, torch.Tensor):
            text_sequence = prompt.to(device)
            if text_sequence.ndim == 1:
                text_sequence = text_sequence.unsqueeze(0).expand(B, -1).contiguous()
        elif isinstance(prompt, list):
            if len(prompt) == 0 or isinstance(prompt[0], int):
                text_sequence = torch.LongTensor([prompt] * B).to(device)
            else:
                text_sequence = torch.LongTensor(prompt).to(device)
        else:
            raise TypeError(f"Unsupported prompt type: {type(prompt)}")

        if beam_width > 1:
            if sample:
                raise ValueError("Beam search does not support sample=True")
            outputs = self._generate_beam_search(
                a_hidden=a_hidden,
                length=length,
                prompt=text_sequence,
                max_generate=max_generate,
                eos_id=eos_id,
                remove_prompt=remove_prompt,
                beam_width=beam_width,
                length_penalty=length_penalty,
                eos_bias=eos_bias,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                return_beam_scores=return_beam_scores,
            )
            if return_encoder_states: outputs['encoder_states'] = {'a_hidden': a_hidden, 'length': length}
            if return_ctc_states: outputs['ctc_states'] = encoder_out['final_posteriors_ctc']
            return outputs

        prompt_length = text_sequence.shape[1]
        if sample is False: temperature = 1.0

        final_seqs = [text_sequence[i].tolist() for i in range(B)]
        final_probs = [[] for _ in range(B)]

        active_orig = list(range(B))
        cur_input = text_sequence
        a_hidden_active = a_hidden
        length_active = length
        cache = None
        steps = 0

        while len(active_orig) > 0 and steps < max_generate:
            decoder_out = self.language_model_decoder(
                tokens = cur_input,
                a_hidden = a_hidden_active,
                a_lengths = length_active,
                cache = cache,
                text_lengths = torch.full(
                    (cur_input.shape[0],), cur_input.shape[1],
                    dtype=torch.long, device=device,
                ),
            )
            logits = decoder_out['logits'][:, -1, :]
            cache = decoder_out['kv_cache']

            probs = (logits / temperature).softmax(dim=-1)
            if sample:
                pred = probs.multinomial(num_samples=1).squeeze(-1)
            else:
                pred = probs.argmax(dim=-1)
            pred_probs = probs.gather(1, pred.unsqueeze(-1)).squeeze(-1)

            steps += 1
            is_eos = (pred == eos_id)
            is_last_step = (steps >= max_generate)

            keep_idx = []
            pred_cpu = pred.tolist()
            eos_cpu = is_eos.tolist()
            prob_cpu = pred_probs.tolist()
            for j, orig_i in enumerate(active_orig):
                final_probs[orig_i].append(prob_cpu[j])
                if eos_cpu[j]:
                    continue  # finalize without appending eos
                final_seqs[orig_i].append(pred_cpu[j])
                if not is_last_step:
                    keep_idx.append(j)

            if is_last_step or len(keep_idx) == 0:
                break

            keep_idx_t = torch.tensor(keep_idx, dtype=torch.long, device=device)
            active_orig = [active_orig[j] for j in keep_idx]
            a_hidden_active = a_hidden_active.index_select(0, keep_idx_t)
            length_active = length_active.index_select(0, keep_idx_t)
            if cache is not None:
                # cache layout: [L, KV=2, B, H, N, D]; batch dim is axis 2.
                cache = {
                    'cache': cache['cache'].index_select(2, keep_idx_t),
                    'cache_lengths': cache['cache_lengths'].index_select(0, keep_idx_t),
                }
            cur_input = pred.index_select(0, keep_idx_t).unsqueeze(1)

        if remove_prompt:
            final_seqs = [seq[prompt_length:] for seq in final_seqs]

        outputs = {'text_sequence': final_seqs, 'probs': final_probs}
        if return_encoder_states: outputs['encoder_states'] = {'a_hidden': a_hidden, 'length': length}
        if return_ctc_states: outputs['ctc_states'] = encoder_out['final_posteriors_ctc']

        return outputs

    def _beam_score(self, score:float, generated_len:int, length_penalty:float) -> float:
        if length_penalty == 0.0:
            return score
        return score / (max(generated_len, 1) ** length_penalty)

    def _tokens_that_repeat_ngram(self, seq:List[int], prompt_length:int, ngram_size:int) -> set:
        generated = seq[prompt_length:]
        if ngram_size <= 0 or len(generated) < ngram_size - 1:
            return set()

        prefix = tuple(generated[-(ngram_size - 1):]) if ngram_size > 1 else tuple()
        banned = set()
        for idx in range(0, len(generated) - ngram_size + 1):
            ngram = tuple(generated[idx:idx + ngram_size])
            if ngram_size == 1 or ngram[:-1] == prefix:
                banned.add(ngram[-1])
        return banned

    @torch.no_grad()
    def _generate_beam_search(
            self,
            a_hidden:torch.Tensor,
            length:torch.Tensor,
            prompt:torch.Tensor,
            max_generate:int,
            eos_id:int,
            remove_prompt:bool,
            beam_width:int,
            length_penalty:float,
            eos_bias:float,
            repetition_penalty:float,
            no_repeat_ngram_size:int,
            return_beam_scores:bool,
        ) -> Dict[str, Union[List[List[int]], List[List[float]], List[float]]]:
        prompt_length = prompt.shape[1]
        final_seqs, final_probs, final_scores = [], [], []

        for row_idx in range(a_hidden.shape[0]):
            row_a_hidden = a_hidden[row_idx:row_idx + 1]
            row_length = length[row_idx:row_idx + 1]
            row_prompt = prompt[row_idx:row_idx + 1]
            beams = [{
                'tokens': row_prompt,
                'seq': row_prompt.squeeze(0).tolist(),
                'probs': [],
                'score': 0.0,
                'cache': None,
                'finished': False,
            }]

            for _ in range(max_generate):
                expanded = []
                for beam in beams:
                    if beam['finished']:
                        expanded.append(beam)
                        continue

                    decoder_out = self.language_model_decoder(
                        tokens=beam['tokens'],
                        a_hidden=row_a_hidden,
                        a_lengths=row_length,
                        cache=beam['cache'],
                        text_lengths=torch.LongTensor([beam['tokens'].shape[1]]).to(row_a_hidden.device),
                    )
                    logits = decoder_out['logits'][:, -1, :]
                    log_probs = logits.log_softmax(dim=-1).squeeze(0)
                    if eos_bias != 0.0:
                        log_probs[eos_id] = log_probs[eos_id] + eos_bias
                    if repetition_penalty != 0.0:
                        generated_tokens = set(beam['seq'][prompt_length:])
                        if len(generated_tokens) > 0:
                            penalty_idx = torch.LongTensor(list(generated_tokens)).to(log_probs.device)
                            log_probs[penalty_idx] = log_probs[penalty_idx] - repetition_penalty
                    banned_tokens = self._tokens_that_repeat_ngram(
                        seq=beam['seq'],
                        prompt_length=prompt_length,
                        ngram_size=no_repeat_ngram_size,
                    )
                    if len(banned_tokens) > 0:
                        banned_idx = torch.LongTensor(list(banned_tokens)).to(log_probs.device)
                        log_probs[banned_idx] = -torch.inf

                    next_log_probs, next_tokens = torch.topk(log_probs, k=min(beam_width, log_probs.shape[-1]))
                    next_probs = next_log_probs.exp()

                    for token, token_log_prob, token_prob in zip(
                        next_tokens.tolist(),
                        next_log_probs.tolist(),
                        next_probs.tolist(),
                    ):
                        is_eos = token == eos_id
                        expanded.append({
                            'tokens': torch.LongTensor([[token]]).to(row_a_hidden.device),
                            'seq': beam['seq'] if is_eos else beam['seq'] + [token],
                            'probs': beam['probs'] + [token_prob],
                            'score': beam['score'] + token_log_prob,
                            'cache': decoder_out['kv_cache'],
                            'finished': is_eos,
                        })

                expanded.sort(
                    key=lambda b: self._beam_score(
                        b['score'],
                        len(b['seq']) - prompt_length,
                        length_penalty,
                    ),
                    reverse=True,
                )
                beams = expanded[:beam_width]
                if all(beam['finished'] for beam in beams):
                    break

            best = max(
                beams,
                key=lambda b: self._beam_score(
                    b['score'],
                    len(b['seq']) - prompt_length,
                    length_penalty,
                ),
            )
            seq = best['seq'][prompt_length:] if remove_prompt else best['seq']
            final_seqs.append(seq)
            final_probs.append(best['probs'])
            final_scores.append(best['score'])

        outputs = {'text_sequence': final_seqs, 'probs': final_probs}
        if return_beam_scores:
            outputs['beam_scores'] = final_scores
        return outputs

    def get_prev_id(self) -> int: return self.language_model_decoder.embed.weight.shape[0] - 1
    def get_pad_id(self) -> int: return 0
    def get_bos_id(self) -> int: return 0
    def get_eos_id(self) -> int: return 0
    def get_first_pass_id(self) -> int: return self.get_bos_id()
    
    def get_blank_id(self) -> int:
        if self.ctc_loss_weight == 0: return None
        return self.ctc_decoder.ff.weight.shape[0] - 1

    @torch.no_grad()
    def transcribe(
            self,
            audio_signal: Union[torch.Tensor, List[torch.Tensor]],
            tokenizer: spm.SentencePieceProcessor,
            previous_text_conditioning: bool = False,
            max_sequence_length: int = -1,
            max_generate: int = 'encoder_states',
            device: str = None,
            verbose=True,
            bos_id=0,
            ctc_history=False,
            synthetic_history=False,
            sample=False,
            temperature=0.2,
            sample_synthetic_history=True,
            temperature_synthetic_history=0.9,
            eval_ctc=False,
            first_pass_ctc=False,
            masked_conditioning=True,
            min_seq_len = -1,
            **kwargs
    ):
        '''
        audio_signal: (B, T, C) | [(B, T, C)]*N
        tokenizer: tokenizer to use for decoding
        previous_text_conditioning: whether to chunk up long format audio and process sequentially like whipser models
        max_sequence_length: maximum sequence length for the model encoder_states means we cap at the size of the encoder states (NOTE: this assumes downsampling is not greater than 8x)
        max_generate: maximum number of tokens to generate -1 means we cap at the size of the encoder states
        device: device to use for the model i.e. 'cuda' or 'cpu' or 'cuda:N'
        verbose: for debugging, prints out generations
        bos_id: beginning of sequence id, this is also used as the eos_id
        ctc_history: whether to use previous ctc output to form the text prompt
        synthetic_history: whether to generate a synthetic history/prompt for the model
        sample: whether to sample from the distribution or take the argmax, this should usually be set to False
        temperature: temperature for sampling
        sample_synthetic_history: whether to sample from the distribution or take the argmax when generating the synthetic history, this should usually be set to True
        temperature_synthetic_history: temperature for sampling the synthetic history
        eval_ctc: whether to return ctc output and not transcribe with the language model
        min_seq_len: pad audio to this size 
        '''
        tensor_input = isinstance(audio_signal, torch.Tensor)
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)
        
        if synthetic_history: assert previous_text_conditioning == False, f'previous_text_conditioning must be False if synthetic_history is True'

        if ctc_history or eval_ctc or first_pass_ctc:
            from lcasr.decoding.greedy import GreedyCTCDecoder
            ctc_decoder = GreedyCTCDecoder(tokenizer=tokenizer, blank_id=self.get_blank_id())
            if ctc_history: assert previous_text_conditioning == True, f'previous_text_conditioning must be True if ctc_history is True'
            else:
                assert previous_text_conditioning == False
                assert ctc_history == False
                assert synthetic_history == False


        if isinstance(audio_signal, torch.Tensor):
            if max_sequence_length == -1: audios = [audio_signal]
            else:
                from lcasr.utils.dataloading import chunk_spectogram
                audios = chunk_spectogram(spec=audio_signal, chunk_size=max_sequence_length)
        
        elif isinstance(audio_signal, list) and all(isinstance(a, torch.Tensor) for a in audio_signal): audios = audio_signal
        else: raise ValueError('audio_signal must be a torch.Tensor or a list of torch.Tensors')

        results = []
        prev_id = self.get_prev_id()
        prompt = None


        for audio in audios:
            if audio.shape[-1] < min_seq_len and min_seq_len > 0: audio = torch.cat((audio, torch.zeros(1, audio.shape[1], min_seq_len-audio.shape[2])), dim=-1)
            audio = audio.to(device)
            print(f'Audio shape: {audio.shape}') if verbose else None
            assert audio.dim() == 3, f'Audio signal must be a 3D tensor (B, T, C) got {audio.dim()}'
            assert audio.shape[0] == 1, f'currently only supports batch size of 1, got {audio.shape[0]}'    

            if eval_ctc: # probably I should just have a seperate function call for this eval..
                output = self.forward(audio_signal=audio)
                ctc_output = output['final_posteriors_ctc'].squeeze(0)
                ctc_text = ctc_decoder(ctc_output, decode=True)
                print(f'Decoded sequence: {ctc_text}') if verbose else None
                results.append(ctc_text.strip())
            else:
                encoder_states = None
                if synthetic_history:
                    output = self.generate(
                        audio_signal = audio,
                        max_generate = 'encoder_states', #30, #'encoder_states',
                        return_encoder_states = True,
                        prompt = [prev_id],
                        bos_id = bos_id,
                        #eos_id=999999999999999,
                        return_ctc_states = False,
                        remove_prompt = False,   
                        sample=sample_synthetic_history,
                        temperature=temperature_synthetic_history,       
                    )         
                    out_sequence = output['text_sequence']
                    encoder_states = output['encoder_states']
                    #out_sequence = [prev_id] + torch.randint_like(torch.tensor(out_sequence), 1, max(out_sequence)).tolist()[1:]

                    print(f'synthetic history: {tokenizer.decode(out_sequence[1:])}') if verbose else None
                    print(out_sequence) if verbose else None
                    prompt = out_sequence + [bos_id]
                elif first_pass_ctc:
                    # self.language_model_decoder.train()
                    # self.language_model_decoder.cross_attn_drop_p = 1.0

                    output = self.forward(audio_signal=audio)
                    ctc_output = output['final_posteriors_ctc'].squeeze(0)
                    ctc_text = ctc_decoder(ctc_output, decode=False) #,sample=True)
                    print(f'CTC output: {tokenizer.decode(ctc_text)}') if verbose else None
                    encoder_states = output

                    if False:
                        ctc_output = ctc_decoder(ctc_output, decode=True)
                        words = ctc_output.split(" ") 
                        # shuffle order
                        random.shuffle(words)
                        ctc_text = " ".join(words)
                        print(f'Shuffled CTC output: {ctc_text}') if verbose else None
                        ctc_text = tokenizer.encode(ctc_text)

                    #if len(ctc_text) > 1: ctc_text = torch.randint_like(torch.tensor(ctc_text), 1, max(ctc_text)).tolist()
                    prompt = [self.get_first_pass_id()] + ctc_text + [bos_id] 

                    # output = self.generate(
                    #     audio_signal = audio,
                    #     max_generate = max_generate,
                    #     return_encoder_states = False,
                    #     prompt = prompt,
                    #     bos_id = bos_id,
                    #     return_ctc_states = ctc_history,
                    #     encoder_states=encoder_states,     
                    #     sample=sample,
                    #     temperature=temperature,    
                    # )
                    # out_sequence = output['text_sequence']
                    # decoded_sequence = tokenizer.decode(out_sequence) 
                    # print(f'First pass output: {decoded_sequence}') if verbose else None
                    # prompt = [self.get_first_pass_id()] + out_sequence + [bos_id]
                elif masked_conditioning:
                    output = self.generate(
                        audio_signal = audio,
                        max_generate = 'encoder_states', #30, #'encoder_states',
                        return_encoder_states = True,
                        prompt = [bos_id],
                        bos_id = bos_id,
                        #eos_id=999999999999999,
                        return_ctc_states = False,
                        remove_prompt = False,   
                        sample=sample,
                        temperature=temperature,       
                    )         
                    out_sequence = output['text_sequence']
                    encoder_states = output['encoder_states']    
                    probs = torch.tensor(output['probs'])
                
                    # get inidices of the smallest n probs
                    k = len(probs) // 5
                    probs = torch.randn_like(probs)
                    min_probs, min_indices = probs.topk(k, largest=False)
         
                    print(f'synthetic history: {tokenizer.decode(out_sequence[1:])}') if verbose else None
                    out_sequence = [0]+ [el if i not in min_indices else 1 for i, el in enumerate(out_sequence[1:])]

                    #out_sequence = [prev_id] + torch.randint_like(torch.tensor(out_sequence), 1, max(out_sequence)).tolist()[1:]

                    print(out_sequence) if verbose else None
                    prompt = out_sequence + [bos_id]
                    

                output = self.generate(
                    audio_signal = audio,
                    max_generate = max_generate,
                    return_encoder_states = False,
                    prompt = prompt,
                    bos_id = bos_id,
                    return_ctc_states = ctc_history,
                    encoder_states=encoder_states,     
                    sample=sample,
                    temperature=temperature,    
                )
                out_sequence = output['text_sequence']
                decoded_sequence = tokenizer.decode(out_sequence) 

                if previous_text_conditioning and not ctc_history: 
                    if False:
                        output = tokenizer.decode(out_sequence)
                        
                        words = output.split(" ") 
                        # shuffle order
                        random.shuffle(words)
                        text = " ".join(words)
                        print(f'Shuffled prev output: {text}') if verbose else None
                        out_sequence = tokenizer.encode(text)

                    prompt = [prev_id] + out_sequence + [bos_id]
                    #if len(out_sequence) > 1: prompt = [prev_id] + torch.randint_like(torch.tensor(out_sequence), 1, max(out_sequence)).tolist() + [bos_id]                
    
                elif ctc_history:
                    ctc_posteriors = output['ctc_states'].squeeze(0)
                    ctc_history_text = ctc_decoder(ctc_posteriors, decode=False)
                    prompt = [prev_id] + ctc_history_text + [bos_id]
                    print(f'CTC output: {ctc_decoder(ctc_posteriors, decode=True)}') if verbose else None


                if verbose: print(f'Decoded sequence: {decoded_sequence}')
                results.append(decoded_sequence.strip())

        if tensor_input:
            results = " ".join(results)

        return results
           
    @torch.no_grad()
    def ctc_beam_search(
        self, 
        audio_signal,
        tokenizer,
        beam_width,
        alpha,
        beta,
        prune_less_than_val,
        top_am_threshold=-6,
    ):
        encoder_out = self.forward(audio_signal=audio_signal)
        a_hidden, length = encoder_out['a_hidden'], encoder_out['length']
        decoder = self.language_model_decoder
        decoder.old_forward = decoder.forward

        def fake_forward(a_hidden, a_length, decoder):
            def fwd(x, length, cache=None):
                cur_a_hidden = a_hidden.clone().expand(x.shape[0], -1, -1)
                cur_a_length = a_length.clone().expand(x.shape[0]) 
        
                decoder_out = decoder.old_forward(tokens = x, a_hidden = cur_a_hidden, a_lengths = cur_a_length, text_lengths = length, cache=cache)
                logits, kv_cache = decoder_out['logits'], decoder_out['kv_cache']
                return logits, None, kv_cache
            return fwd
        
        decoder.forward = fake_forward(a_hidden, length, decoder)
        
        language_model = ctc_beam_search.LanguageModel(
            model = decoder,
            bos_id = 0,
            device = a_hidden.device,
        )

        beamsearch = ctc_beam_search.BeamSearch(
            tokenizer=tokenizer,
            beam_width=beam_width,
            log_probs=encoder_out['final_posteriors_ctc'][0].clone().to('cpu'),
            alpha=alpha,
            beta=beta,
            prune_less_than_val=prune_less_than_val,
            top_am_threshold=top_am_threshold,
            language_model=language_model,
            blank_id=len(tokenizer),
            debug=False
        )
        beamsearch.run_search()
        decoder.forward = decoder.old_forward
        return beamsearch.return_text(idx = 0)

    def forward(
            self, 
            audio_signal,
            text_sequence = None, 
            length = None, 
            cache: Dict = None,
            return_logits = False
        ):

        max_audio_length: int = audio_signal.size(-1)
        cached_kvs = None
        
        if max_audio_length < self.min_seq_len and self.min_seq_len > 0:
            audio_signal = torch.cat((
                audio_signal,
                torch.zeros(
                    audio_signal.size(0),
                    audio_signal.size(1),
                    self.min_seq_len - audio_signal.size(2),
                    device=audio_signal.device,
                )
            ), dim=-1)
            max_audio_length: int = audio_signal.size(-1)

        if length is None or self.no_encoder_padding:
            length = torch.tensor([max_audio_length] * audio_signal.size(0), device=audio_signal.device)
            
        audio_signal = torch.transpose(audio_signal, 1, 2)
        audio_signal, length = self.subsampling(audio_signal, lengths = length) if not self.checkpoint_subsampling else checkpoint(self.create_custom_forward(self.subsampling), audio_signal, length)
        #audio_signal = self.abs_pos_enc(audio_signal, scale = self.pos_enc_scale)
        max_audio_length = audio_signal.size(1)
        ## create masks
        
        mask = torch.arange(max_audio_length, device=audio_signal.device).expand(audio_signal.size(0), max_audio_length) >= length.unsqueeze(1)
    
        rotary_emb_fn = None
   
        full_kv_lengths = length 
        if self.use_rotary:
            max_seq_len = full_kv_lengths.max()
            q_offset = 0 if cached_kvs is None else cached_kvs.shape[1]
      
            cos, sin = self.rotary_pos_emb(max_seq_len, audio_signal.device)
            rotary_emb_fn = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = self.rotary_pos_emb.learned_freq)
        

        if length.max() == length.min():
            att_mask, mask = None, None
        else:
            full_kv_mask = torch.arange(full_kv_lengths.max(), device=audio_signal.device).expand(audio_signal.size(0), full_kv_lengths.max()) >= full_kv_lengths.unsqueeze(1)
            if audio_signal.device.type == 'cuda' and self.flash_attn:
                att_mask = ~full_kv_mask
            else:
                qmask, kmask = ~mask, ~full_kv_mask
                att_mask = ~(rearrange(qmask, 'b n -> b () n ()') * rearrange(kmask, 'b n -> b () () n'))
                att_mask = att_mask.to(audio_signal.dtype) * -torch.finfo(audio_signal.dtype).max

        pad_mask = mask 
    
        audio_signal = self.pos_enc(audio_signal)

        for lth, layer in enumerate(self.layers):

            if self.checkpoint_every_n_layers > 0 and lth % self.checkpoint_every_n_layers == 0:
                audio_signal, _ = checkpoint(
                    self.create_custom_forward(layer), 
                    audio_signal, # x
                    att_mask, # att_mask
                    pad_mask, # pad_mask
                    length,
                    None,
                    self.flash_attn,
                    rotary_emb_fn
                )
    
            else:
                audio_signal, _ = layer(
                    x = audio_signal, 
                    attn_mask = att_mask, 
                    pad_mask = pad_mask,
                    length = length,
                    cached_kv = None,
                    flash_attn = self.flash_attn,
                    rotary_emb_fn = rotary_emb_fn
                )
            
            if lth != len(self.layers) - 1 and self.self_conditioning:
                iterim_post = torch.nn.functional.softmax(self.ctc_decoder(x=audio_signal, logits=True), dim=-1)
                audio_signal = self.ctc_decoder.integrate_projections(audio_signal, self.ctc_decoder.project_back(iterim_post))        
        
        final_posts_ctc = None
        if self.ctc_loss_weight > 0:
            final_posts_ctc = self.ctc_decoder(x = self.ctc_decoder.norm(audio_signal), logits = return_logits) 

        final_posts_lm, kv_cache = None, None
        if text_sequence is not None:
            lm_out = self.language_model_decoder(
                tokens = text_sequence,
                a_hidden = audio_signal,
                a_lengths = length,
                cache = cache,
            )
            final_posts_lm = lm_out['logits']
            kv_cache = lm_out['kv_cache']


        return {
            'final_posteriors_ctc': final_posts_ctc,
            'final_posteriors_lm': final_posts_lm,
            'a_hidden': audio_signal,
            'length': length,
            'kv_cache': kv_cache,
        }
    

 

class ConformerLayer(nn.Module):
    def __init__(
        self,
        d_model,
        conv_kernel_size,
        dropout_ff,
        dropout_conv,
        dropout_attn,
        layer_idx,
        total_layers,
        head_dim,
        n_heads,
        default_norm = DEFAULT_NORM,
        sandwich_norm = False,
        bias_in_ff = True,
        transformer = False,
        conv_expansion_factor = 1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.total_layers = total_layers
        self.sandwich_norm = sandwich_norm
        self.bias_in_ff = bias_in_ff
        self.trasformer = transformer

        
        if not self.trasformer:
            self.conv = PreNorm(
                d_model = d_model, 
                fn = ConformerConvolution(
                    d_model = d_model,
                    kernel_size = conv_kernel_size,
                    norm_type = kwargs.get('conv_norm', 'batch_renorm'),
                    exp_factor = conv_expansion_factor,
                ),
                norm = default_norm
            )
            self.do_conv = nn.Dropout(dropout_conv)

        if not self.trasformer:
            self.ff1 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff), norm = default_norm, sandwich_norm = sandwich_norm))
        
        self.ff2 = Scale(0.5, PreNorm(d_model = d_model, fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff), norm = default_norm, sandwich_norm = sandwich_norm))
        self.do_ff = nn.Dropout(dropout_ff)

        self.attend = PreNorm(
            d_model = d_model, 
            fn = Attention(
                n_feats = d_model,
                head_dim = head_dim,
                n_heads = n_heads,
                dropout = dropout_attn,
                bias = False,
                layer_idx = layer_idx,
                **kwargs
            ),
            norm = default_norm,
        )
        self.attn_norm_out = default_norm(d_model) if sandwich_norm else lambda x: x

        self.do_attn_out = nn.Dropout(min(dropout_ff, 0.1)) # don't wan't this too large
        self.norm_out = default_norm(d_model)

            

    def forward(self, x, attn_mask, pad_mask, length, cached_kv = None, flash_attn = True, rotary_emb_fn = None):
        '''
        pad_mask: mask for padding used in conv layers
        attn_mask: attn_mask this should include the cached keys and values
        length: list of lengths of the input sequence
        cached_kv: kvs from previous block-reccurrent time step
        '''

        if not self.trasformer:
            x = self.do_ff(self.ff1(x)) + x

        attn_out = self.attend(
            x = x,
            length = length,
            attn_mask = attn_mask,
            pad_mask = pad_mask,
            flash_attn = flash_attn,
            rotary_emb_fn = rotary_emb_fn
        )
        x = self.attn_norm_out(self.do_attn_out(attn_out)) + x
        
        if not self.trasformer:
            x = self.do_conv(self.conv(x, pad_mask = pad_mask)) + x
    
        x = self.do_ff(self.ff2(x)) + x

        x = self.norm_out(x)

        return x, None
    



class CrossAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        bias=False,
        dropout=0.0,
        **kwargs
    ):
        super().__init__()
        self.layer_idx = kwargs.get('layer_idx', None)
        self.flash_attn = kwargs.get('flash_attn', True)

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
   
        self.activation = nn.Softmax(dim=-1)

        self.dropout_p = dropout
        self.causal = False
        # softmax_scale is set to None but will default to 1/sqrt(d_k) in FlashAttention
        self.flash_attn_c_fn = FlashCrossAttention(softmax_scale = None, attention_dropout = dropout, causal = False)

        self.qkv_proj = nn.Linear(n_feats, 3 * n_heads * head_dim, bias=bias)
        self.q_proj = nn.Linear(n_feats, n_heads * head_dim, bias=bias)
        self.kv_proj = nn.Linear(n_feats, 2 * n_heads * head_dim, bias=bias)

        self.kv = lambda x: rearrange(self.kv_proj(x), "b n (h d kv) -> kv b n h d", kv=2, h=n_heads, d=head_dim)
        self.q = lambda x: rearrange(self.q_proj(x), "b n (h d) -> b n h d", h=n_heads, d=head_dim)

        self.out_proj = nn.Linear(n_heads * head_dim, n_feats, bias=bias)

    @staticmethod
    def apply_rotary(q, kv, rotary_emb_fn): 
        if rotary_emb_fn is not None:
            if rotary_emb_fn.learned == False:
                q, kv[:, :, 0] = rotary_emb_fn.apply(q, kv[:, :, 0])
            else:
                k, v = kv[:, :, 0], kv[:, :, 1]
                q, k = rotary_emb_fn.apply(q, k)
                kv = torch.stack([k, v], dim=2)
        return q, kv
        
    def forward(self, xq, xkv, kv_mask = None, attn_mask=None, rotary_emb_fn = None):
        H, D = self.n_heads, self.head_dim

        flash_attn = self.flash_attn

        q = self.q(xq)
        k, v = self.kv(xkv)
        kv = torch.stack([k, v], dim=2)

        q, kv = self.apply_rotary(q, kv, rotary_emb_fn)

        ### Flash attention stuff 
        if xq.device.type == 'cuda' and flash_attn:
            q, kv = q.contiguous(), kv.contiguous()
            if q.dtype == torch.float32:
                q, kv = q.half(), kv.half()

            if kv_mask is None:
                out = self.flash_attn_c_fn(q, kv)
            else:
                q_attn_mask = torch.ones((q.shape[0], q.shape[1]), device=q.device).bool()
                kv_attn_mask = kv_mask

                b, qs, qh, qd = q.shape
                b, kvs, kvn, kh, kd = kv.shape
                q_up, q_indices, cu_seq_lens, max_seqlen, _ = unpad_input(q, q_attn_mask)
                kv_up, kv_indices, k_cu_seq_lens, max_k_seq_len, _ = unpad_input(kv, kv_attn_mask)
                
                out = self.flash_attn_c_fn(
                    q_up, 
                    kv_up, 
                    cu_seqlens = cu_seq_lens.to(torch.int32),
                    max_seqlen = max_seqlen,
                    cu_seqlens_k = k_cu_seq_lens.to(torch.int32),
                    max_seqlen_k = max_k_seq_len,
                )
                out = pad_input(out, indices = q_indices, batch = b, seqlen = qs)
            out = out.to(xq.dtype) 
            out = rearrange(out, "b n h d -> b n (h d)")
        else:
            k, v = rearrange(kv, "b n kv h d -> kv b h n d", kv=2).contiguous()
            q = q.transpose(1, 2).contiguous()
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=0.0 if not self.training else self.dropout_p, is_causal=False)
            out = rearrange(out, "b h n d -> b n (h d)")


        out = self.out_proj(out)
        
        return out

def l2norm(t, groups=1, dim=-1):
    if groups == 1:
        return F.normalize(t, p=2, dim=dim)
    t = rearrange(t, '... (g d) -> ... g d', g=groups)
    t = F.normalize(t, p=2, dim=dim)
    return rearrange(t, '... g d -> ... (g d)')


# def get_flex_attention_score_function(pos_bias, causal=True):
#     assert causal, 'Not Implemented'
#     def score_mod(score, b, h, q_idx, kv_idx):
#         return torch.where(q_idx >= kv_idx, score * pos_bias[:, h, q_idx, kv_idx], -torch.finfo(score.dtype).max)
#     return score_mod


class CosineAttention(nn.Module):
    def __init__(
        self,
        n_feats,
        head_dim,
        n_heads,
        dropout=0.1,
        bias=False,
        cosine_sim=True,
        temperature=15.5,
        return_attention=False,
        causal=True,
        **kwargs
    ):
        super().__init__()
        self.shared_kv = kwargs.get('shared_kv', False)
        # 'none', 'pre', 'both', 'post'
        self.talking_heads = kwargs.get('talking_heads', 'none')

        self.n_feats, self.head_dim, self.n_heads = n_feats, head_dim, n_heads
        self.dropout = nn.Dropout(dropout)
        self.bias = bias
        self.return_attention = return_attention
        self.causal = causal

        self.cosine_sim = cosine_sim

        if self.talking_heads == 'pre' or self.talking_heads == 'both': self._head_proj = nn.Conv2d(n_heads, n_heads, (1, 1))
        if self.talking_heads == 'post' or self.talking_heads == 'both': self._head_proj_post = nn.Conv2d(n_heads, n_heads, (1, 1))

        self.use_sdpa = kwargs.get('decoder_use_sdpa', True)
        if self.use_sdpa:  assert self.talking_heads == 'none', 'sdpa not compatible with talking heads'


        self.temperature = torch.nn.Parameter(torch.tensor(
            temperature), requires_grad=True) if isinstance(temperature, float) else temperature

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
        if mode == 'pre' and (self.talking_heads == 'pre' or self.talking_heads == 'both'): dots = self._head_proj(dots)
        if mode == 'post' and (self.talking_heads == 'post' or self.talking_heads == 'both'): dots = self._head_proj_post(dots)
        return dots

    def attend(self, query, key, value, attn_mask, pos_bias): 
        if not self.use_sdpa:
            dots = einsum('bhid,bhjd->bhij', query, key) * self.temperature
            dots = self.head_proj(dots, mode='pre')
            
            dots += pos_bias.masked_fill_(attn_mask, -torch.finfo(dots.dtype).max)

            attn = self.activation(dots)
            attn = self.head_proj(attn, mode='post')

            attn = self.dropout(attn)
            return torch.matmul(attn, value)
        else:
            query = query * self.temperature # apply before sdpa so gradient is computed
            pos_bias = pos_bias.masked_fill(attn_mask, -torch.finfo(pos_bias.dtype).max) # apply pos_bias as an additive mask by combining with attn_mask
            out = F.scaled_dot_product_attention(
                query=query,
                key=key,
                value=value,
                attn_mask=pos_bias,
                is_causal=False,
                dropout_p=0.0 if not self.training else self.dropout.p,
                scale=1.0, # scale is already applied to query
                enable_gqa=self.shared_kv,
            )
            return out
    


    @staticmethod
    def attach_cache(kv, cache, cache_indices):
        kv = torch.stack(kv, dim=0)
        if cache is None: return kv
        if exists(cache_indices):
            zero_vector = torch.zeros_like(kv[:, :, :, :1, :])
            kv_w_cache = torch.cat([cache, kv, zero_vector], dim=-2)
            # we do this to remove unnecessary padding
            kv_w_cache = torch.gather(kv_w_cache, dim=-2, index=cache_indices)
        else: kv_w_cache = torch.cat([cache, kv], dim=-2)
        return kv_w_cache

    def forward(self, x, pos_bias, mask, cache=None, cache_indices=None):
        B, N, C, H, D = *x.shape, self.n_heads, self.head_dim

        q, k, v = self.qkv(x)
        q, k = map(l2norm, (q, k)) if self.cosine_sim else (q, k)
        kv = self.attach_cache([k, v], cache, cache_indices)
        k, v = kv

        out = self.attend(q, k, v, mask, pos_bias)

        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.out_proj(out)
        return out, kv


class CrossAttnDecoder(nn.Module):
    def __init__(
        self,
        vocab_size = 4096,
        n_layers = 3,
        d_model = 768,
        n_heads = 6,
        head_dim = 128,
        expansion_factor = 4,
        dropout_ff = 0.0,
        dropout_attn = 0.0,
        decoder_norm = True,
        rotary_interpolation_factor = 1.0, # https://arxiv.org/abs//2306.15595 Extending Context Window of Large Language Models via Positional Interpolation
        default_norm = 'rms_norm',
        bias_in_ff = False,
        **kwargs
    ):
        super().__init__()


        self.d_model = d_model
        n_layers = n_layers if kwargs.get('decoder_layers', None) is None else kwargs.get('decoder_layers')
        self.n_layers = n_layers 
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.expansion_factor = expansion_factor
        self.dropout_ff = dropout_ff
        self.dropout_attn = dropout_attn
        self.decoder_norm = decoder_norm
        self.rotary_interpolation_factor = rotary_interpolation_factor
        self.bias_in_ff = bias_in_ff
        self.default_norm = default_norm
        self.flash_attn = kwargs.get('flash_attn', True)

        self.cross_attn_drop_p = kwargs.get('cross_attn_drop_p', 0.0)
        self.cross_attn_kv_drop_p = kwargs.get('cross_attn_kv_drop_p', 0.0)

        additional_embeddings = kwargs.get('additional_embeddings', 0)
        self.embed = nn.Embedding(vocab_size + additional_embeddings, d_model)
        self.abs_pos_dec = kwargs.get('use_abs_pos_dec', True)
        self.pos_enc = LearnableFourierPosEnc(d_model, hidden_dim=kwargs.get('fourier_pos_hidden_dim', 64)) if self.abs_pos_dec else nn.Identity()

        self.use_rotary_cross_attn = kwargs.get('rotary_cross_attn', False)
        self.rotary_pos_emb = None
        if self.use_rotary_cross_attn:
            self.rotary_pos_emb = RotaryPositionalEmbedding(
                dim = head_dim,
                base = 1500000,
                learned_freq = False,
            )

        self.dropout_emb = kwargs.get('dropout_emb', 0.0)
        self.ff_out_dropout = kwargs.get('ff_out_dropout', 0.0)
        self.causal = True
        accepted_norms = ['rms_norm', 'layer_norm']
        assert default_norm in accepted_norms, f'default_norm must be one of {accepted_norms} (got {default_norm})'
        default_norm = RMSNorm if default_norm == 'rms_norm' else LayerNorm
        self.acoustic_norm = default_norm(d_model) if kwargs.get('acoustic_norm', False) else nn.Identity()

        self.learnt_cache_size = kwargs.get('learnt_cache_size', 0)
        if self.learnt_cache_size > 0:
            self.learnt_cache = nn.Embedding(self.learnt_cache_size, d_model)
            self.cache_merge = nn.Sequential(
                nn.Linear(head_dim*n_layers, head_dim*n_layers),
            )

        self.layers = nn.ModuleList([])
        for _ in range(n_layers):
            self.layers.append(nn.ModuleList([
                PreNorm(
                    d_model = d_model,
                    fn = CosineAttention(
                        n_feats = d_model,
                        n_heads = n_heads,
                        head_dim = head_dim,
                        bias = bias_in_ff,
                        causal = True,
                        dropout = dropout_attn,
                        temperature = kwargs.get('decoder_attention_temperature', 15.5),
                        **kwargs
                    ),
                    norm = default_norm
                ),
                PreNorm(
                    d_model = d_model, 
                    fn = CrossAttention(
                        n_feats = d_model,
                        n_heads = n_heads,
                        head_dim = head_dim,
                        bias = bias_in_ff,
                        dropout = dropout_attn,
                        **kwargs
                    ), 
                    norm = default_norm
                ),
                PreNorm(
                    d_model = d_model, 
                    fn = ConformerFeedForward(d_model, bias1 = bias_in_ff, bias2 = bias_in_ff),
                    norm = default_norm,
                )
            ]))

        self.cache_needs_gather = False

        self.out_proj = nn.Sequential(
            default_norm(d_model) if decoder_norm else nn.Identity(),
            nn.Linear(d_model, vocab_size)
        )

        self.positional_bias = DynamicPositionBias(
            dim=64,
            heads=n_heads,
            depth=2,
            log_distance=False,
        )

    @staticmethod
    def get_cache(cache, layer):
        if cache is None: return None
        else: return cache['cache'][layer]
    
    

    @staticmethod
    def get_cache_indices(x_lens, cache_lens, cache_kv, x):
        # used later w/ gather to remove padding when cache is concatenated with current input to remove padding
        max_new_len = (x_lens + cache_lens).max()
        # cache kv =  LAYERS, KEYS+VALUES (2), BATCH, HEADS, N, DIM
        B, H, N, D = x.shape[0], cache_kv.shape[-3], (x.shape[1] +
                                                      cache_kv.shape[-2]), cache_kv.shape[-1]
        indices = []
        for i in range(B):  # stinky for loop to sort out indices for gather
            cache_indices = torch.arange(cache_lens[i], device='cpu')
            total_length = cache_lens[i] + x_lens[i]
            diff_from_max_len = max_new_len - total_length
            x_indices = torch.arange(
                x_lens[i]+diff_from_max_len, device='cpu') + cache_kv.shape[-2]
            if diff_from_max_len > 0:
                # last index will be used for padding
                x_indices[-diff_from_max_len:] = N
            new_indices = torch.cat([cache_indices, x_indices])
            indices.append(new_indices)

        indices = torch.stack(indices, dim=0)

        # 2 for key and value
        indices = rearrange(
            indices, 'b n -> () b () n ()').expand(2, B, H, -1, D)
        return indices.to(x.device)

    def create_masks_and_positions(self, x, length, cache):
        ''' We do this so kv caching can be done when there is padding in the kv cache'''
        x_len = length if length is not None else torch.tensor(
            x.shape[-2], device=x.device).expand(x.shape[0])
        cache_len = cache['cache_lengths'] if exists(cache) else 0

        total_len = x_len + cache_len
        kv_mask = torch.arange(total_len.max(), device=x.device).expand(
            len(total_len), -1) >= total_len.unsqueeze(-1)
        q_mask = torch.arange(x_len.max(), device=x.device).expand(
            len(x_len), -1) >= x_len.unsqueeze(-1)
        attn_mask = ~(rearrange(~q_mask, "b n -> b () n ()") *
                      rearrange(~kv_mask, "b n -> b () () n"))
        ##
        ##
        causal_mask = repeat(torch.arange(
            total_len.max(), device=x.device), 'i -> b r i', b=len(total_len), r=x_len.max())
        cache_offset = cache_len[:, None, None] if exists(cache) else cache_len
        diagonal_offset = torch.arange(x_len.max(), device=x.device)[
            None, :, None]
        ##
        ## positional stuff ##
        positional_grid = (causal_mask - cache_offset - diagonal_offset) * -1
        pos = torch.arange(positional_grid.min(), positional_grid.max(
        )+1, device=x.device, dtype=x.dtype)[:, None]
        min_cache_len = 0 if cache_len.__class__ == int else cache_len.min()
        # shift so zero is the smallest number
        positional_indices = ((positional_grid) +
                              (total_len.max() - min_cache_len - 1))
        pos_bias = self.positional_bias(
            pos=pos, indices=positional_indices, dtype=x.dtype, device=x.device)
        ## positional stuff ##
        ##
        if self.causal:
            causal_mask = causal_mask >= (cache_offset + diagonal_offset + 1)
            attn_mask = torch.logical_or(attn_mask, causal_mask[:, None])
        ##
        return q_mask, attn_mask, total_len, x_len, cache_len, pos_bias

    def create_learnt_cache(self, batch_size, a_hidden, a_lengths):
        x = self.learnt_cache.weight.unsqueeze(0).expand(batch_size, -1, -1)
        lengths = torch.LongTensor([x.shape[1]] * batch_size).to(x.device)
        mask, attn_mask, total_lens, x_len, cache_len, pos_bias = self.create_masks_and_positions(x, lengths, None)
        cache_indices = None

        if a_lengths.max() == a_lengths.min(): kv_mask = None # if all the same length don't bother with the mask
        else: kv_mask = ~(torch.arange(a_hidden.shape[1], device=a_hidden.device).expand(a_hidden.size(0), a_hidden.shape[1]) >= a_lengths.unsqueeze(1))

        cross_attn_mask = None
        if (a_hidden.device.type != 'cuda' or not self.flash_attn) and kv_mask is not None: # create attention mask
            #kv_mask = kv_mask if kv_mask is not None else ~(torch.arange(a_hidden.shape[1], device=a_hidden.device).expand(a_hidden.size(0), a_hidden.shape[1]) >= a_lengths.unsqueeze(1))
            q_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device) # no mask
            cross_attn_mask = ~(rearrange(~q_mask, 'b n -> b () n ()') * rearrange(~kv_mask, 'b n -> b () () n'))

        rotary_emb_fn = None
        if self.use_rotary_cross_attn:
            max_seq_len = a_hidden.shape[-2] + x.shape[-2]
            q_offset = a_hidden.shape[-2]
            cos, sin = self.rotary_pos_emb(max_seq_len, a_hidden.device)
            rotary_emb_fn = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = False, trim_k=True)

        kv_cache = []

        for lth, (self_attn, cross_attn, ff_out) in enumerate(self.layers):
            z, kv = self_attn(
                x = x,
                pos_bias = pos_bias, 
                mask = attn_mask,
                cache = self.get_cache(None, lth),
                cache_indices = cache_indices
            )
            x = x + z
            kv_cache.append(kv)
            x = cross_attn(
                x, 
                xkv = a_hidden, 
                kv_mask = kv_mask, 
                attn_mask = cross_attn_mask,
                rotary_emb_fn = rotary_emb_fn,
            ) + x
            x = F.dropout(ff_out(x), p=self.ff_out_dropout, training=self.training) + x


        kv_cache = torch.stack(kv_cache, dim=0) 
        ## cache kv =  LAYERS, KEYS+VALUES (2), BATCH, HEADS, N, DIM
        kv_cache = rearrange(kv_cache, 'l kv b h n d -> kv b h n (l d)', l=self.n_layers, kv=2)

        kv_cache = self.cache_merge(kv_cache)
        kv_cache = rearrange(kv_cache, 'kv b h n (l d) -> l kv b h n d', l=self.n_layers, kv=2)
        kv_cache = {'cache_lengths': total_lens, 'cache': kv_cache}
        return kv_cache

    def forward(
            self,
            tokens: torch.Tensor, 
            a_hidden: torch.Tensor, 
            a_lengths: torch.Tensor,
            text_lengths: torch.Tensor = None,
            cache:Dict=None,
        ):
        '''
        tokens: (batch, seq_len) - target text sequence
        a_hidden: (batch, seq_len, dim) - encoder output
        '''
        lengths = torch.LongTensor([tokens.shape[1]] * tokens.shape[0]).to(tokens.device) if text_lengths is None else text_lengths
        offsets = cache['cache_lengths'] if exists(cache) else None

        if offsets != None and self.learnt_cache_size > 0: offsets = offsets - self.learnt_cache_size

        if tokens.ndim == 2: x = self.embed(tokens)
        else: x = tokens; assert tokens.ndim == 3, 'tokens must be 2D or 3D tensor if already embedded'

        if self.abs_pos_dec: x = self.pos_enc(x, lengths=lengths, position_offsets=offsets)
        x = F.dropout(x, p=self.dropout_emb, training=self.training)
        a_hidden = self.acoustic_norm(a_hidden)
        

        rotary_emb_fn = None
        if self.use_rotary_cross_attn:
            assert offsets is None, 'not implemented with cache yet'
            max_seq_len = a_hidden.shape[-2] + tokens.shape[-1]
            q_offset = a_hidden.shape[-2]
            cos, sin = self.rotary_pos_emb(max_seq_len, a_hidden.device)
            rotary_emb_fn = apply_rotary(cos = cos, sin = sin, q_offset = q_offset, learned = False, trim_k=True)

        if not exists(cache) and self.learnt_cache_size > 0: cache = self.create_learnt_cache(a_hidden.shape[0], a_hidden, a_lengths)


        mask, attn_mask, total_lens, x_len, cache_len, pos_bias = self.create_masks_and_positions(x, lengths, cache)
        cache_indices = self.get_cache_indices(x_len, cache_len, cache['cache'], x) if exists(cache) and self.cache_needs_gather else None


        if a_lengths.max() == a_lengths.min() and self.cross_attn_kv_drop_p > 0.0: kv_mask = None # if all the same length don't bother with the mask
        else: kv_mask = ~(torch.arange(a_hidden.shape[1], device=a_hidden.device).expand(a_hidden.size(0), a_hidden.shape[1]) >= a_lengths.unsqueeze(1))

        #print(kv_mask)

        cross_attn_mask = None
        if (a_hidden.device.type != 'cuda' or not self.flash_attn) and kv_mask is not None: # create attention mask
            #kv_mask = kv_mask if kv_mask is not None else ~(torch.arange(a_hidden.shape[1], device=a_hidden.device).expand(a_hidden.size(0), a_hidden.shape[1]) >= a_lengths.unsqueeze(1))
            q_mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=x.device) # no mask
            cross_attn_mask = ~(rearrange(~q_mask, 'b n -> b () n ()') * rearrange(~kv_mask, 'b n -> b () () n'))

        kv_cache = []

        # self.training = True
        # self.cross_attn_drop_p = 100.
        if self.training and self.cross_attn_drop_p > 0.0: drop_probs = torch.rand(a_hidden.shape[0], device=a_hidden.device) < self.cross_attn_drop_p


        for lth, (self_attn, cross_attn, ff_out) in enumerate(self.layers):
            z, kv = self_attn(
                x = x,
                pos_bias = pos_bias, 
                mask = attn_mask,
                cache = self.get_cache(cache, lth),
                cache_indices = cache_indices
            )
            x = x + z
            kv_cache.append(kv)
            cross_attn_out = cross_attn(
                x, 
                xkv = a_hidden, 
                kv_mask = kv_mask, 
                attn_mask = cross_attn_mask,
                rotary_emb_fn = rotary_emb_fn,
            ) 

            if self.training and self.cross_attn_drop_p > 0.0: cross_attn_out = torch.masked_fill(cross_attn_out, drop_probs[:, None, None], 0.0)

            x = x + cross_attn_out
            x = F.dropout(ff_out(x), p=self.ff_out_dropout, training=self.training) + x


        kv_cache = torch.stack(kv_cache, dim=0) if len(kv_cache) > 0 else None
        kv_cache = {'cache_lengths': total_lens, 'cache': kv_cache} if exists(kv_cache) else None
        self.cache_needs_gather = x_len.max() != x_len.min()

        return {'logits':self.out_proj(x), 'kv_cache':kv_cache}


class RLEncDecSconformerV2(EncDecSconformerV2):

    @torch.no_grad()
    def batch_generate_rl_prompt___(
        self,
        encoder_states:Dict[str, torch.Tensor], # encoder states
        temperature:float = 1.0, # temperature for sampling
        to_generate:int = 20, # number of tokens to generate
    ):
        a_hidden, length = encoder_states['a_hidden'], encoder_states['length']
        batch_size = a_hidden.shape[0]
        prompt = torch.LongTensor([[self.get_prev_id()] for _ in range(batch_size)]).to(a_hidden.device)
        cache = None
        generated = 0

        generated_sequence = prompt.clone()

        while generated < to_generate:
            decoder_out = self.language_model_decoder(
                tokens = prompt,
                a_hidden = a_hidden,
                a_lengths = length,
                cache = cache,
                text_lengths = torch.tensor([prompt.shape[1]]).to(a_hidden.device),
            )
            decoder_logits = decoder_out['logits']
            cache = decoder_out['kv_cache']
            decoder_pred = (decoder_logits[:, -1, :] / temperature).softmax(dim=-1).multinomial(num_samples=1)
            generated_sequence = torch.cat([generated_sequence, decoder_pred], dim=1)
            prompt = decoder_pred
            generated += 1

        return generated_sequence

    def calc_loss__(
            self, 
            audio_signal, # B, C, T
            text_sequence,
            a_lengths,
            t_lengths,
            lm_text_sequence=None,
            lm_text_sequence_lengths=None,
            bos_id=0, 
            eos_id=0,
            encoder_outputs=None,
            tokenizer=None,
            **kwargs
        ):
        if encoder_outputs is None:
            encoder_outputs = self(audio_signal, length=a_lengths)


        if lm_text_sequence is None: # add bos to text sequence
            text_sequence_bos = F.pad(text_sequence, (1, 0), value=bos_id)
            target_lengths_bos = t_lengths + 1
        else:
            assert lm_text_sequence_lengths is not None, 'lm_text_sequence_lengths must be provided if lm_text_sequence is provided'
            text_sequence_bos = lm_text_sequence
            target_lengths_bos = lm_text_sequence_lengths

        
        ctc_out = encoder_outputs['final_posteriors_ctc']
        a_length_out = encoder_outputs['length']
        
        lm_out = self.language_model_decoder(
            tokens = text_sequence_bos,
            a_hidden = encoder_outputs['a_hidden'],
            a_lengths = encoder_outputs['length'],
            cache = None,
        )
        lm_out = lm_out['logits']

        if self.ctc_loss_weight > 0.0:
            if kwargs.get('trim_ctc_by', None) is not None:
                trim_ctc_by = kwargs.get('trim_ctc_by', None)
                a_lengths = a_lengths - trim_ctc_by
                in_length = audio_signal.shape[-1]
                out_length = ctc_out.shape[1]
                downsample_factor = in_length / out_length
                trim_ctc_by = int(round(trim_ctc_by / downsample_factor))
                a_length_out = a_length_out - trim_ctc_by
                
            else: trim_ctc_by = 0
            ctc_loss = F.ctc_loss(
                log_probs = rearrange(ctc_out[:, trim_ctc_by:], 'b n c -> n b c'),
                targets = text_sequence,
                input_lengths = a_length_out,
                target_lengths = t_lengths,
                reduction = 'sum',
                blank = ctc_out.shape[-1] - 1
            )

            a_sum = a_lengths.sum()
            ctc_loss_to_show = (ctc_loss / a_sum).item() * 100
            ctc_loss_to_bwd = ctc_loss / (ctc_out[:, trim_ctc_by:].shape[1] * ctc_out.shape[0]) * 100
        else:
            ctc_loss_to_show, ctc_loss_to_bwd = 0, 0


        targets = text_sequence_bos.clone()
        targets[:, :-1] = text_sequence_bos[:, 1:]
        if target_lengths_bos.max() == target_lengths_bos.min(): targets[:, -1] = 0
        else:
            targets = add_eos(targets, eos_id = eos_id, token_lens = target_lengths_bos)
        mask = token_lens_to_mask(target_lengths_bos)
        targets = mark_padding(targets, mask, pad_id = -100)
        
        lm_num_masked = 0
        if lm_loss_mask is not None:
            assert lm_loss_mask.shape == targets.shape, f'lm_loss_mask shape {lm_loss_mask.shape} does not match targets shape {targets.shape}'
            lm_loss_mask = lm_loss_mask.to(targets.device)
            targets = targets.masked_fill(lm_loss_mask, -100)
            lm_num_masked = lm_loss_mask.sum()

            
        predictions = lm_out
        lm_loss = F.cross_entropy(
            input = rearrange(predictions, 'b n c -> (b n) c'),
            target = rearrange(targets, 'b n -> (b n)'),
            ignore_index = -100,
            reduction = 'none'
        )
        if kwargs.get('return_token_losses', False):
            lm_loss = rearrange(lm_loss, '(b n) -> b n', b = predictions.shape[0])
            other_outputs['lm_loss'] = lm_loss
            other_outputs['lm_loss_mask'] = lm_loss_mask

        lm_loss = lm_loss.sum()
        lm_loss_to_show = (lm_loss / (target_lengths_bos.sum() - lm_num_masked)).item() 
        lm_loss_to_bwd = lm_loss / ((predictions.shape[0] * predictions.shape[1]) - lm_num_masked) 

        loss_to_show = ctc_loss_to_show * self.ctc_loss_weight + lm_loss_to_show * (1 - self.ctc_loss_weight)
        loss = ctc_loss_to_bwd * self.ctc_loss_weight + lm_loss_to_bwd 

        wandb_log_data = {
            'loss': loss_to_show,
            'ctc_loss': ctc_loss_to_show,
            'lm_loss': lm_loss_to_show,
        }

        return {
            'loss': loss,
            'display_losses': wandb_log_data,
            'ctc_posteriors': ctc_out,
            'lm_posteriors': lm_out,
            'length': a_length_out,
            **other_outputs
        }


    @staticmethod # for converting enc dec for RL training
    def reformat_checkpoint(
        checkpoint_path:str,
        save_path:str,
    ):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        checkpoint['config']['model_class'] = "RLEncDecSconformerV2"
        checkpoint['config']['training']['loss_on_previous'] = False
        checkpoint['config']['training']['condition_on_previous'] = False

        torch.save(checkpoint, save_path)
        print(f"Checkpoint reformatted and saved to {save_path}")

    @staticmethod
    def match_vectors_to_embeddings(embedding_weight: torch.Tensor, vectors: torch.Tensor) -> torch.Tensor:
        """
        Matches each vector in `vectors` to the nearest index in the `embedding_weight` tensor
        using cosine similarity.

        Args:
            embedding_weight (torch.Tensor): Tensor of shape (num_embeddings, embedding_dim)
            vectors (torch.Tensor): Tensor of shape (batch_size, embedding_dim)

        Returns:
            torch.Tensor: Tensor of shape (batch_size,) containing the indices of the nearest embeddings
        """
        # Normalize both embeddings and vectors to unit vectors for cosine similarity
        embedding_norm = F.normalize(embedding_weight, p=2, dim=1)  # (num_embeddings, dim)
        vector_norm = F.normalize(vectors, p=2, dim=1)              # (batch_size, dim)

        # Compute cosine similarity: (batch_size, num_embeddings)
        similarity = torch.matmul(vector_norm, embedding_norm.T)

        # Take the index of the highest similarity (closest match)
        nearest_indices = similarity.argmax(dim=1)

        return nearest_indices

    def transcribe(
            self,
            audio_signal: Union[torch.Tensor, List[torch.Tensor]],
            tokenizer: spm.SentencePieceProcessor,
            targets: List[str] = None,
            previous_text_conditioning: bool = False,
            max_sequence_length: int = -1,
            max_generate: int = 'encoder_states',
            device: str = None,
            verbose=True,
            bos_id=0,
            ctc_history=False,
            synthetic_history=False,
            sample=False,
            temperature=0.2,
            sample_synthetic_history=True,
            temperature_synthetic_history=0.9,
            eval_ctc=False,
            first_pass_ctc=False,
            min_seq_len = -1
    ):
        tensor_input = isinstance(audio_signal, torch.Tensor)
        assert not tensor_input
        if device == None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.to(device)
        
        elif isinstance(audio_signal, list) and all(isinstance(a, torch.Tensor) for a in audio_signal): audios = audio_signal
        else: raise ValueError('audio_signal must be a torch.Tensor or a list of torch.Tensors')

        results = []
        prev_id = self.get_prev_id()
        prompt = None


        for i, audio in enumerate(audios):
            if audio.shape[-1] < min_seq_len and min_seq_len > 0: audio = torch.cat((audio, torch.zeros(1, audio.shape[1], min_seq_len-audio.shape[2])), dim=-1)
            audio = audio.to(device)
            print(f'Audio shape: {audio.shape}') if verbose else None
            assert audio.dim() == 3, f'Audio signal must be a 3D tensor (B, T, C) got {audio.dim()}'
            assert audio.shape[0] == 1, f'currently only supports batch size of 1, got {audio.shape[0]}'    

            if targets is not None:
                target = targets[i]
                #print(f'Target: {target}') if verbose else None
                int_target = tokenizer.encode(target)
                #print(f'Int target: {int_target}') if verbose else None
                prompt = [bos_id] + int_target + [bos_id]
            

                # prev_embed = self.language_model_decoder.embed(torch.LongTensor([bos_id]).to(audio.device).unsqueeze(0))
                # bos_embed = self.language_model_decoder.embed(torch.LongTensor([bos_id]).to(audio.device).unsqueeze(0))
                # average_emedding = self.language_model_decoder.embed.weight.mean(dim=0, keepdim=True).unsqueeze(0)
                # average_emedding = average_emedding.repeat(1, 40, 1)
                # prompt_sequence = average_emedding

                # a_hidden = None
                # for i in range(2):
                #     prompt = torch.cat([prev_embed, prompt_sequence, bos_embed], dim=1)
                #     prompt = nn.Parameter(prompt, requires_grad=True)
                #     target_embedding = self.language_model_decoder.embed(torch.LongTensor(int_target).to(audio.device).unsqueeze(0))
                #     text_input = torch.cat([prompt, target_embedding], dim=1)

                #     target_sequence = int_target + [bos_id]
                #     target_sequence_length = len(target_sequence)
                #     target_sequence = torch.LongTensor(target_sequence).to(audio.device).unsqueeze(0)
                    
                #     if a_hidden is None:
                #         out = self.forward(audio, text_input)
                #         a_hidden = out['a_hidden']
                #         preds = out['final_posteriors_lm'][:, -target_sequence_length:, :]
                #     else:
                #         out = self.language_model_decoder(
                #             tokens= text_input,
                #             a_hidden = a_hidden,
                #             a_lengths = torch.tensor([audio.shape[1]]).to(audio.device),
                #         )
                #         preds = out['logits'][:, -target_sequence_length:, :]

                #     loss = F.cross_entropy(
                #         input = preds.reshape(-1, preds.shape[-1]),
                #         target = target_sequence.reshape(-1),
                #         ignore_index = -100,
                #         reduction = 'mean'
                #     )
                #     # get gradient w.r.t. the prompt
                #     grad = torch.autograd.grad(loss, prompt, retain_graph=False)[0]
                    
                    
                #     prompt_sequence = prompt[:,1:-1, :].detach().clone()
                #     prompt_sequence = prompt_sequence - grad[:, 1:-1, :].detach() * 5
            
                # nearest_ = self.match_vectors_to_embeddings(self.language_model_decoder.embed.weight, prompt_sequence.reshape(-1, prompt_sequence.shape[-1])).tolist()
                # nearest = []
                # for el in nearest_:
                #     if el != 0: nearest.append(el)
                #     else: break
                # try:
                #     print(f'decoded nearest: {tokenizer.decode(nearest)}') if verbose else None
                # except:
                #     print(f'failed to decode nearest: {nearest}') if verbose else None
                # #print(nearest)
      
                # prompt = [bos_id] + nearest + [bos_id]
                
                # print(self.language_model_decoder.embed(prompt).shape) if verbose else None
            
            #prompt = [prev_id] + tokenizer.encode("hello how are you? not bad okay thats fine i dont mind whatever") + [bos_id]
            output = self.generate(
                audio_signal = audio,
                max_generate = max_generate,
                return_encoder_states = False,
                prompt = prompt,
                bos_id = bos_id,
                return_ctc_states = ctc_history,
                encoder_states=None,     
                sample=sample,
                temperature=temperature,    
            )
            out_sequence = output['text_sequence']
            decoded_sequence = tokenizer.decode(out_sequence) 


            if verbose: print(f'Decoded sequence: {decoded_sequence}')
            results.append(decoded_sequence.strip())


        return results


       

import argparse
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--save_path', type=str, default=None)

    args = parser.parse_args()
    if args.checkpoint is not None:
        RLEncDecSconformerV2.reformat_checkpoint(
            checkpoint_path=args.checkpoint,
            save_path=args.save_path,
        )
    else:
        # run debug tests

        model = EncDecSconformerV2(decoder_use_sdpa=False, shared_kv=True) # default model
        model.print_total_params()
        device = 'cuda'
        vocab_size = 4096
        audio_seq = torch.randn(2, 80, 100, device=device)
        text = torch.randint(0, vocab_size, (2, 10), device=device)
        model.to(device)
        model.eval()
        print(text.shape, audio_seq.shape)

        out = model(audio_seq, text)
        print('final_posteriors_ctc', out['final_posteriors_ctc'].shape)
        print('final_posteriors_lm', out['final_posteriors_lm'].shape)
        print('a_hidden', out['a_hidden'].shape)
        print('length', out['length'].shape)
        print('kv_cache', out['kv_cache']['cache'].shape)
        print('kv_cache', out['kv_cache']['cache_lengths'])

        print(model.calc_loss(
            audio_signal = audio_seq,
            text_sequence = text,
            a_lengths = torch.tensor([100, 100], device=device),
            t_lengths = torch.tensor([5, 10], device=device)
        )['loss'])


        new_model = EncDecSconformerV2(decoder_use_sdpa=True, shared_kv=True)
        new_model.load_state_dict(model.state_dict())
        model = new_model
        model.to(device)
        model.eval()
        sdpa_out = model(audio_seq, text)

        print(out['final_posteriors_lm'][0,0])
        print('--')
        print(sdpa_out['final_posteriors_lm'][0,0])
        print(
            torch.allclose(
                out['final_posteriors_lm'], 
                sdpa_out['final_posteriors_lm'],
                rtol=1e-3, atol=1e-3, # tolerance used in flash attn github tests
            )
        )
        print(
            torch.allclose(
                out['a_hidden'], 
                sdpa_out['a_hidden'],
            )
        )

        ### RL model

        model = RLEncDecSconformerV2()
        device = 'cuda'
        vocab_size = 4096   
        audio_seq = torch.randn(5, 80, 200, device=device) 
        text = torch.randint(0, vocab_size, (5, 15), device=device)
        model.to(device)
        model.eval()

        from lcasr.utils.audio_tools import load_tokenizer
        tokenizer = load_tokenizer()

        loss = model.calc_loss(
            audio_signal=audio_seq, 
            text_sequence=text, 
            a_lengths=torch.tensor([200, 200, 200, 200, 200], device=device),
            t_lengths=torch.tensor([5, 15, 15, 15, 15], device=device),
            tokenizer=tokenizer,
        )
        print(loss['display_losses'])






                    
