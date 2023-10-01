import torch, numpy as np, pickle as pkl, os
from typing import List, Dict, Tuple, Optional, Union
from importlib import reload as rl
from tqdm import tqdm
from lming.utils.helpers import exists
from lming.models.transformer import transformer_lm
import time, heapq
from multiprocessing import Pool
import math
from functools import partial

from einops import rearrange, repeat


class Beam():
    def __init__(
        self, 
        state,
        am_sequence = [],
        lm_sequence = [],
        stimes = [],
        next_lm_token_lps = None,
        score = 0,
    ):
        '''
        state: lm state containing k,v cache 
        am_sequence: sequence of tokens from the acoustic model
        lm_sequence: sequence of tokens from the language model
        stimes: start positions of the tokens in the lm_sequence
        '''
        self.state = state 
        self.score = score
        self.am_sequence = am_sequence
        self.lm_sequence = lm_sequence
        self.next_lm_token_lps = next_lm_token_lps
        self.stimes = stimes

    def __str__(self):
        return f"{self.am_sequence}"

    def __repr__(self):
        return self.__str__()


class LanguageModel():
    def __init__(
        self, 
        model:transformer_lm, 
        bos_id=0, 
        device='cpu', 
        temperature=1.0,
        half_precision=False,
        keep_states_on_device=True,
    ):
        self.model = model
        self.model.eval()
        self.bos_id = bos_id
        self.device = torch.device(device)
        self.model.to(self.device)
        self.half = half_precision
        if self.half:
            self.model.half()
        self.temperature = temperature
        self.offload_device = self.device if keep_states_on_device else 'cpu'
    
    def logits_to_lprobs(self,logits):
        logits = logits  / self.temperature if self.temperature != 1.0 else logits  
        return logits.log_softmax(dim=-1)

    @torch.no_grad()
    def get_initial_state(self):
        x, length = torch.tensor([[self.bos_id]]).to(self.device), torch.LongTensor([1]).to(self.device)
        logits, _, state = self.model(x=x, length=length)
        return self.logits_to_lprobs(logits).squeeze().to('cpu'), {k:v.to(self.offload_device) for k,v in state.items()}

    @staticmethod
    def move_to_device(input_ids, input_lengths, states, device):
        input_ids, input_lengths = input_ids.to(device), input_lengths.to(device)
        states = {k:v.to(device) for k,v in states.items()} if exists(states) else None
        return input_ids, input_lengths, states


    @torch.no_grad()
    def __call__(self, input_ids, input_lengths, states=None):
        input_ids, input_lengths, states = self.move_to_device(input_ids, input_lengths, states, self.device)
        logits, _, new_state = self.model(x=input_ids, length=input_lengths, cache=states)
        return self.logits_to_lprobs(logits).to('cpu'), {k:v.to(self.offload_device) for k,v in new_state.items()}


class BeamSearch(): 
    def __init__(
            self, 
            tokenizer, 
            beam_width, 
            log_probs, 
            language_model: LanguageModel,
            alpha=0.4, # score = am + lm*alpha + beta if not blank or repitition else am
            beta=0.4, # beta offsets for lack of blank and repitition in lm probs
            blank_id=128,
            blank_penalty=0.0, # additional penalty for blank
            repitition_penalty=0.0, # additional penalty for repitition
            top_am_threshold=-6, # top am scores to consider
            max_cache_length = -1, # max length of cache to keep in memory
            debug=False,
            prune_less_than_val=None,
            cache_init=None
        ):
        self.tokenizer = tokenizer
        self.beam_width = beam_width
        self.vocab_size = tokenizer.vocab_size()
        self.log_probs = log_probs
        self.language_model = language_model
        self.blank_id = blank_id
        self.alpha = alpha
        self.beta = beta
        self.beams = []
        self.position = 0 # position in sequence
        self.blank_penalty = blank_penalty
        self.max_cache_length = max_cache_length
        self.repitition_penalty = repitition_penalty
        self.top_am_threshold = top_am_threshold
        self.debug = debug
        self.prune_less_than_val = prune_less_than_val
        self.cache_init = cache_init

    def initiate(self):
        assert len(self.beams) == 0 and self.position == 0, 'initiate can only be called once | beams should be empty'
        lm_logps, state = self.language_model.get_initial_state()
        state = state if self.cache_init is None else self.cache_init
        self.beams = [
            Beam(
                state = state,
                am_sequence = [None], # no bos for am
                lm_sequence = [self.language_model.bos_id], # bos for lm
                next_lm_token_lps = lm_logps, # log probs of next token
                stimes = [self.position], # start time
            )
        ]

    def return_text(self, idx):
        if idx >= len(self.beams):
            print('Beam index out of range')
            return
        beam = self.beams[idx]
        return self.tokenizer.decode(beam.lm_sequence[1:])


    def print_beams(self):
        for i, beam in enumerate(self.beams):
            print(f'{i}: {self.return_text(i)} | {beam.score}')

    def prune(self, beams):
        print(f'Num beams to sort: {len(beams)}') if self.debug else None
        beams = heapq.nlargest(self.beam_width, beams, key=lambda beam: beam.score) # faster than sort
        return beams

    @staticmethod
    def _sum_log_scores(s1:float, s2:float) -> float:
        return s1+math.log(1+math.exp(s2-s1)) if s1>=s2 else s2+math.log(1+math.exp(s1-s2))

    def merge(self, beams):
        self.beam_dict = {}
        for beam in beams:
            beam_str = str(beam)
            if beam_str in self.beam_dict:
                self.beam_dict[beam_str].score = self._sum_log_scores(beam.score, self.beam_dict[beam_str].score)
            else:
                self.beam_dict[beam_str] = beam
        return list(self.beam_dict.values())

            
    @staticmethod
    def trim_cache(state, new_length): 
        amount_to_trim = state['cache_lengths'][-1] - new_length
        if amount_to_trim <= 0:
            return state
        state['cache'] = state['cache'][:, :, :, :, amount_to_trim:, :]
        state['cache_lengths'] = state['cache_lengths'] - amount_to_trim 
        return state

    def grab_state(self, state, indice):
        ## TODO: trim cache to be equal to cache_lengths
        cache_len = state['cache_lengths'][indice]
        #print(state['cache'].shape)
        cache = {
            'cache': state['cache'][:,:,indice, None], 
            'cache_lengths': state['cache_lengths'][indice, None],
        }
        cache['cache'] = cache['cache'][:, :, :, :, :cache_len, :]
        cache = self.trim_cache(cache, self.max_cache_length) if self.max_cache_length > 0 else cache
        return cache

    def run_search(self, use_tqdm=True):
        search = True
        pbar = tqdm(total=len(self.log_probs)) if use_tqdm else None
        pbar.update(self.position) if use_tqdm else None
        while search:
            search = self.step()
            pbar.update(1) if use_tqdm else None
        pbar.close() if use_tqdm else None

    def prune_less_than(self, beams):
        if self.prune_less_than_val is None:
            return
        top_beam_score = beams[0].score
        num_beams = len(beams)
        beams = [beam for beam in beams if not beam.score < (top_beam_score - self.prune_less_than_val)] # prune beams that are less than top beam score by prune_less_than_val
        post_prune_num_beams = len(beams)
        print(f'Pruned {num_beams - post_prune_num_beams} beams') if self.debug else None
        return beams

    def step(self):
        if self.position == len(self.log_probs):
            return False
        if self.position == 0 and len(self.beams) == 0:
            self.initiate()

        # now create new beams
        new_beams = []
        
        cur_am_lgps = self.log_probs[self.position]
        
        # only look at top k am scores 
        #top_am_indices = torch.topk(cur_am_lgps, self.top_am_threshold, dim=0, sorted=False, largest=True).indices
        top_am_indices = torch.arange(cur_am_lgps.shape[-1])[(cur_am_lgps > (cur_am_lgps[cur_am_lgps.argmax()] + self.top_am_threshold))].tolist()
        #print(top_am_indices.shape, len(self.beams))
        stime = time.time_ns()
        operations = 0
        blank_operations = 0
        for beam in self.beams: # kinda sloww
            beam_lm_probs = beam.next_lm_token_lps
            
            #beam_lm_probs[top_am_indices-1] = torch.nn.functional.log_softmax(beam_lm_probs[top_am_indices-1], dim=-1)
            beam_lm_probs = beam_lm_probs * self.alpha + self.beta
            #joint_am_lm_probs = (beam_lm_probs + cur_am_lgps[1:-1]) # joint_am_lm_probs[i-1] + beam.score
            # top_am_indices_lm = np.array(top_am_indices)
            # top_am_indices_lm = top_am_indices_lm[top_am_indices_lm != self.blank_id]
            # top_am_indices_lm = top_am_indices_lm[top_am_indices_lm != beam.am_sequence[-1]]
            # if len(top_am_indices_lm) > 0:
            #     beam_lm_probs[top_am_indices_lm] = beam_lm_probs[top_am_indices_lm] - beam_lm_probs[top_am_indices_lm].logsumexp(dim=-1, keepdim=True)

            for i in range(1, self.vocab_size+1):
                if i not in top_am_indices:
                    continue
                
                operations += 1
                b_am_seq, b_lm_seq, b_stimes = beam.am_sequence, beam.lm_sequence, beam.stimes
                #stime = time.time()
                
                if b_am_seq[-1] == i or i == self.blank_id: # won't need scoring from language model
                    blank_operations += 1
                    
                    new_beam = Beam(
                        state = beam.state,
                        am_sequence = b_am_seq + [i] if i == self.blank_id and b_am_seq[-1] != self.blank_id else b_am_seq,
                        lm_sequence = b_lm_seq,
                        stimes = b_stimes,
                        next_lm_token_lps = beam.next_lm_token_lps,
                        score = self.log_probs[self.position][i] + beam.score + (self.blank_penalty if i == self.blank_id else self.repitition_penalty)
                    )
                else:
                    new_beam = Beam(
                        state = beam.state,
                        am_sequence = b_am_seq + [i] if b_am_seq[-1] != self.blank_id else b_am_seq[:-1] + [i], # remove blank if it is followed by a non blank
                        lm_sequence = b_lm_seq + [i], 
                        stimes = b_stimes + [self.position],
                        next_lm_token_lps = None, # will be updated in next step
                        score = self.log_probs[self.position][i] + beam_lm_probs[i] + beam.score
                    )

                   
             
                new_beams.append(new_beam)
          
        etime = time.time_ns()
        print('beam time', (etime - stime)/1e9, 'operations: ',operations, 'blank operations: ', blank_operations) if self.debug else None

        new_beams = self.prune_less_than(self.prune(self.merge(new_beams))) # merge beams, prune up to top k, prune less than top beam score by prune_less_than_val

        if self.position == len(self.log_probs) - 1: # exit if we are at the end 0:
            self.beams = new_beams
            return False
        
        states, state_lens, sequences, sequence_lens = [], [], [], []
        #L, KV, _, H, N, D = new_beams[0].state['cache'].shape
     
        for beam in new_beams: # already fast
            if beam.next_lm_token_lps is None:
                states.append(rearrange(beam.state['cache'], 'l kv b h n d -> n (l kv b h) d')) # so we can use rnn.pad_sequence
                #print(states[-1].shape, beam.state['cache'].shape)
                sequences.append(torch.tensor(beam.lm_sequence[-1]))
                sequence_lens.append(1)
                state_lens.append(beam.state['cache_lengths'])
    

        if len(sequences) != 0:
            L, KV, _, H, N, D = new_beams[0].state['cache'].shape # Layer, Key+Value, Batch, Head, Sequence, Dimension
            stime = time.time()
            states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True, padding_value=0)
            states = rearrange(states, 'nb n (l kv b h) d -> l kv (b nb) h n d', l=L, kv=KV, h=H, d=D)
            sequences = torch.stack(sequences, dim=0)[:, None]
            sequence_lens = torch.LongTensor(sequence_lens)
            states = {'cache': states, 'cache_lengths': torch.cat(state_lens, dim=0)}

            print(f'batch size: {sequences.shape[0]}') if self.debug else None
            logps, states = self.language_model(input_ids=sequences, input_lengths=sequence_lens, states=states)
            logp_idx = 0
            for beam in new_beams:
                if beam.next_lm_token_lps is None:
                    beam.next_lm_token_lps = logps[logp_idx][-1]
                    beam.state = self.grab_state(states, logp_idx)
                    logp_idx += 1
        
            etime = time.time()
            #print('LM time: ', etime-stime) if self.debug else None

        self.beams = new_beams
        self.position += 1
        return True