import argparse
from omegaconf import OmegaConf
import os

from operator import attrgetter
import random
import subprocess

SAVE_DIR = './.tmp'

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.') # split the string by the last dot
    if pre: # if there is a prefix, get the nested attribute
        obj = rgetattr(obj, pre)
    setattr(obj, post, val) # set the attribute with the new value

def rgetattr(obj, attr):
    pre, _, post = attr.partition('.') # split the string by the first dot
    if post: # if there is a suffix, get the nested attribute
        return rgetattr(getattr(obj, pre), post)
    return getattr(obj, pre) # return the final attribute

def main(args):
    template = OmegaConf.load(args.template)
    copies = [OmegaConf.create({k:template[k].copy() for k in template['template_info']['include_keys']}) for i in range(template['template_info']['create'])]

    for i in range(len(copies)):
        for template_key in template['template_info']['template_keys']:
            val_to_set = attrgetter(template_key)(copies[i])[i]
            rsetattr(copies[i], template_key, val_to_set)

    names = [f'{i}_{random.randint(0,1000000)}' for i in range(len(copies))]

    if not os.path.exists(SAVE_DIR):
        os.mkdir(SAVE_DIR)

    for i in range(len(copies)):
        OmegaConf.save(copies[i], os.path.join(SAVE_DIR, f'{names[i]}.yaml'))
        run_string = f"""#!/bin/bash\n
#SBATCH --time=96:00:00
#SBATCH --mem=170GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module unload CUDA/11.7.0
module unload cuDNN/8.4.1.50-CUDA-11.7.0
module load Anaconda3/2022.10 binutils/2.31.1-GCCcore-8.2.0 CUDA/11.8.0 cuDNN/8.6.0.163-CUDA-11.8.0 GCCcore/8.2.0
source activate /mnt/parscratch/users/acp21rjf/env/h100/

python train_xl.py -config {os.path.join(SAVE_DIR, f'{names[i]}.yaml')} -num_workers 0 -rm_sched -reset_step""" # -debug_hooks
        with open(os.path.join(SAVE_DIR, f'{names[i]}.sh'), 'w') as f:
            f.write(run_string)
        subprocess.run(['sbatch', os.path.join(SAVE_DIR, f'{names[i]}.sh')])
        print(f'Launched {names[i]}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-template','--template', type=str, required=True, help='Path to the template config file')
    args = parser.parse_args()
    main(args)
