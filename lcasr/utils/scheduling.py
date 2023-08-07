import torch, numpy as np

class CosineLRScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, peak_value, final_value):
        self.is_warmup = True
        self.warmup_steps = warmup_steps
        self.peak_value = peak_value
        self.final_value = final_value
        super().__init__(optimizer)
        
    def is_warming_up(self):
        if self.is_warmup:
            return self.last_epoch < self.warmup_steps
        else:
            return False

    def set_cosine_schedule(self, remaining_steps):
        # reset the step to 0
        self.last_epoch = 0
        self.is_warmup = False
        self.steps = remaining_steps

    def get_lr(self):
        if self.is_warmup:
            return [self.peak_value * min(1.0, self.last_epoch / self.warmup_steps) for _ in self.base_lrs]
        else:
            return [self.final_value + 0.5 * (self.peak_value - self.final_value) * (1 + np.cos((self.last_epoch) / (self.steps) * np.pi)) for _ in self.base_lrs]

        

class SequenceWarmupManager():
    def __init__(
            self,
            increase_every:int,
            stop_after:int,
            start_after:int,
            initial_sequence_length:int,
            initial_batch_size:int,
            max_sequence_length:int,
            increase_by_multiplier:float = 2.0,
            batch_size_multiplier:float = 0.5,
            cur_position:int = 0,
            steps_since_last_increase:int = 0,
    ):
        self.increase_every = increase_every
        self.stop_after = stop_after
        self.start_after = start_after
        self.initial_sequence_length = initial_sequence_length
        self.max_sequence_length = max_sequence_length
        self.increase_by_multiplier = increase_by_multiplier
        self.cur_position = cur_position
        self.batch_size_multiplier = batch_size_multiplier

        self.cur_sequence_length = initial_sequence_length
        self.cur_batch_size = initial_batch_size
        self.steps_since_last_increase = steps_since_last_increase

    def __call__(self, steps = 1):
        self.cur_position += steps
        if self.cur_position < self.start_after:
            return False, self.cur_sequence_length, self.cur_batch_size
        elif self.cur_position >= self.stop_after:
            return False, self.cur_sequence_length, self.cur_batch_size
        elif self.cur_sequence_length * self.increase_by_multiplier > self.max_sequence_length:
            return False, self.cur_sequence_length, self.cur_batch_size

        self.steps_since_last_increase += steps
        if self.steps_since_last_increase >= self.increase_every:
            self.steps_since_last_increase = 0
            self.cur_sequence_length = int(self.cur_sequence_length * self.increase_by_multiplier)
            self.cur_batch_size = int(self.cur_batch_size * self.batch_size_multiplier)
            return True, self.cur_sequence_length, self.cur_batch_size
        else:
            return False, self.cur_sequence_length, self.cur_batch_size