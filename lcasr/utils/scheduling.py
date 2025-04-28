import torch, numpy as np
from typing import List
import random

class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, peak_value, final_value):
        self.is_warmup = True
        self.warmup_steps = warmup_steps
        self.peak_value = peak_value
        self.final_value = final_value
        self.offset = 0
        super().__init__(optimizer)
        
    def is_warming_up(self):
        if self.is_warmup:
            return self.last_epoch < self.warmup_steps
        else:
            return False

    def set_cosine_schedule(self, total_recordings, cur_podcast):
        # reset the step to 0
        self.last_epoch = 0
        self.is_warmup = False
        self.steps = total_recordings - cur_podcast + 1
        self.offset = -cur_podcast

    def get_lr(self):
        if self.is_warmup:
            return [self.peak_value * min(1.0, self.last_epoch / self.warmup_steps) for _ in self.base_lrs]
        else:
            return [self.final_value + 0.5 * (self.peak_value - self.final_value) * (1 + np.cos((self.last_epoch + self.offset) / (self.steps) * np.pi)) for _ in self.base_lrs]

        

class SequenceWarmupManager():
    def __init__(
            self,
            increase_every:int, # set to -1 to disable
            stop_after:int,
            start_after:int,
            initial_sequence_length:int,
            initial_batch_size:int,
            max_sequence_length:int,
            increase_by_multiplier:float = 2.0,
            batch_size_multiplier:float = 0.5,
            cur_position:int = 0,
            steps_since_last_increase:int = 0,
            **kwargs
    ):
        self.increase_every = increase_every
        self.stop_after = stop_after
        self.start_after = start_after
        
        self.max_sequence_length = max_sequence_length
        self.increase_by_multiplier = increase_by_multiplier
        self.cur_position = cur_position
        self.batch_size_multiplier = batch_size_multiplier

        self.cur_sequence_length = initial_sequence_length
        self.cur_batch_size = initial_batch_size
        self.steps_since_last_increase = steps_since_last_increase

    def step(self, steps = 1):
        if self.increase_every == -1: # disabled
            return False, self.cur_sequence_length, self.cur_batch_size
        next_seq_len = max(int(self.cur_sequence_length * self.increase_by_multiplier), 1)
        
        self.cur_position += steps
        if self.cur_position < self.start_after:
            return False, self.cur_sequence_length, self.cur_batch_size

        elif self.cur_position >= self.stop_after and self.steps_since_last_increase < self.increase_every/2:
            return False, self.cur_sequence_length, self.cur_batch_size

        elif self.cur_sequence_length * self.increase_by_multiplier > self.max_sequence_length:
            if self.cur_sequence_length != self.max_sequence_length:
                next_seq_len = self.max_sequence_length
            else:
                return False, self.cur_sequence_length, self.cur_batch_size

        elif self.cur_position >= self.stop_after and self.steps_since_last_increase >= self.increase_every/2: # double
            self.steps_since_last_increase = 0
            self.cur_sequence_length = next_seq_len
            self.cur_batch_size = max(int(self.cur_batch_size * self.batch_size_multiplier), 1)
            return True, self.cur_sequence_length, self.cur_batch_size

        self.steps_since_last_increase += steps
        if self.steps_since_last_increase >= self.increase_every:
            self.steps_since_last_increase = 0
            self.cur_sequence_length = next_seq_len
            self.cur_batch_size = max(int(self.cur_batch_size * self.batch_size_multiplier), 1)
            return True, self.cur_sequence_length, self.cur_batch_size
        else:
            return False, self.cur_sequence_length, self.cur_batch_size

    def state_dict(self): # return all state variables
        return self.__dict__

    def load_state_dict(self, state_dict): # load all state variables
        self.__dict__.update(state_dict)



class RandomSequenceLengthManager():
    def __init__(
            self,
            start_after:int,
            initial_sequence_length:int,
            initial_batch_size:int,
            sequence_lengths:List[int] = [512, 1024, 2048, 3072],
            batch_sizes:List[int] = [352, 176, 88, 44],
            cur_position:int = 0,
            **kwargs
    ):

        self.start_after = start_after
        self.sequence_lengths = sequence_lengths
        
        assert isinstance(self.sequence_lengths, list) and len(self.sequence_lengths) > 0, f"sequence_lengths must be a non-empty list got: {self.sequence_lengths}"
        assert isinstance(self.batch_sizes, list) and len(self.batch_sizes) > 0, f"batch_sizes must be a non-empty list got: {self.batch_sizes}"
        assert len(self.sequence_lengths) == len(self.batch_sizes), f"sequence_lengths and batch_sizes must have the same length, got: {len(self.sequence_lengths)} and {len(self.batch_sizes)}"
        assert isinstance(sequence_lengths[0], int), f"sequence_lengths must be a list of integers, got: {self.sequence_lengths}"
        assert isinstance(batch_sizes[0], int), f"batch_sizes must be a list of integers, got: {self.batch_sizes}"
        
        self.cur_position = cur_position
        self.batch_sizes = batch_sizes

        self.cur_sequence_length = initial_sequence_length
        self.cur_batch_size = initial_batch_size


    def step(self, steps = 1):        
        self.cur_position += steps
        
        if self.cur_position < self.start_after:
            return False, self.cur_sequence_length, self.cur_batch_size
        else:
            seq_idx = random.choice(list(range(len(self.sequence_lengths))))
            new_sequence_length = self.sequence_lengths[seq_idx]
            if new_sequence_length == self.cur_sequence_length:
                return False, self.cur_sequence_length, self.cur_batch_size
            else:
                new_batch_size = self.batch_sizes[seq_idx]
                self.cur_sequence_length = new_sequence_length
                self.cur_batch_size = new_batch_size
                return True, self.cur_sequence_length, self.cur_batch_size


    def state_dict(self): # return all state variables
        return self.__dict__

    def load_state_dict(self, state_dict): # load all state variables
        self.__dict__.update(state_dict)