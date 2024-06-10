import json
import os
import subprocess
import torch
import re
from lcasr.models.sconformer_xl import SCConformerXL


class LcasrPreTrainedModel():
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrainedmodels.
    """
    def __init__(self, model):
        self.model = model

    def forward(self, spec, *args, **kwargs):
        return self.model(spec, *args, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name="lcasr-80s",
                        download=False,
                        config=None,
                        device='cpu',
                        repeat=None, # will load prefix+.pt or repeat 1 by default
                        checkpoint_prefix="step_105360"
                      ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            pass
        else:
            hf_url = f'https://huggingface.co/rjflynn2/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

        checkpoints_in_path = [f for f in os.listdir(pretrained_model_name_or_path) if f.startswith(checkpoint_prefix) and f.endswith('.pt')]
        if repeat == None:
            cp_name = checkpoint_prefix + '.pt'
            cp_name = cp_name if cp_name in checkpoints_in_path else f'{checkpoint_prefix}_repeat_1.pt'
            assert cp_name in checkpoints_in_path, f'checkpoint {cp_name} not found in {pretrained_model_name_or_path}'
        else:
            cp_name = f'{checkpoint_prefix}_repeat_{repeat}.pt'
            assert cp_name in checkpoints_in_path, f'checkpoint {cp_name} not found in {pretrained_model_name_or_path}'
        pretrained_model_name_or_path = os.path.join(pretrained_model_name_or_path, cp_name)

        loaded_ckpt = torch.load(
            pretrained_model_name_or_path,
            map_location=torch.device(device)
        )
        if config == None:
            config = loaded_ckpt['config']
        weights = loaded_ckpt['model']
        model = SCConformerXL(vocab_size=4095, **config.model)
        model.load_state_dict(weights)
        return cls(model)
        

      


