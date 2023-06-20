from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset
from typing import List, Optional, Iterator
import torch, math

class lcasrDistributedSampler(DistributedSampler):
    def __init__(
        self, 
        dataset: Dataset, 
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None, 
        shuffle: bool = True,
        seed: int = 0, 
        drop_last: bool = False
        ) -> None:
        super().__init__(self, dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __iter__(self) -> Iterator[List[int]]:
        # see https://pytorch.org/docs/stable/_modules/torch/utils/data/distributed.html#DistributedSampler
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[:padding_size]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
        
    # If len stays the same you can leave it out, else you can also modify it
    #def __len__(self):
        #return self.num_samples