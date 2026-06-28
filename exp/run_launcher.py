import argparse
from omegaconf import OmegaConf
import os

from operator import attrgetter
import random
import re
import subprocess

SAVE_DIR = './.tmp'

run_strings = {
    'a100':f"""#!/bin/bash\n
#SBATCH --time=80:00:00
#SBATCH --mem=82GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

""",
    'h100':f"""#!/bin/bash\n
#SBATCH --time=60:00:00
#SBATCH --mem=100GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=16

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main/

""",
    'h100nvl':f"""#!/bin/bash\n
#SBATCH --time=90:00:00
#SBATCH --mem=130GB
#SBATCH --partition=gpu-h100-nvl
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate /mnt/parscratch/users/acp21rjf/conda/main

"""
}

SBATCH_OPTION_NAMES = {
    'time': 'time',
    'mem': 'mem',
    'partition': 'partition',
    'gres': 'gres',
    'qos': 'qos',
    'cpus_per_task': 'cpus-per-task',
}


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

def sanitize_name(name):
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name)).strip('_')[:160]

def set_sbatch_option(lines, option, value):
    if value is None:
        return lines
    prefix = f'#SBATCH --{option}'
    directive = f'#SBATCH --{option}={value}'
    for ix, line in enumerate(lines):
        if line.startswith(prefix):
            lines[ix] = directive
            return lines
    insert_at = 1 if lines and lines[0].startswith('#!') else 0
    lines.insert(insert_at, directive)
    return lines

def build_run_string(args, config_path, run_name):
    lines = run_strings[args.mode].splitlines()

    for arg_name, sbatch_name in SBATCH_OPTION_NAMES.items():
        lines = set_sbatch_option(lines, sbatch_name, getattr(args, arg_name))

    if args.job_name_prefix:
        lines = set_sbatch_option(lines, 'job-name', sanitize_name(f'{args.job_name_prefix}-{run_name}'))
    if args.log_dir:
        os.makedirs(args.log_dir, exist_ok=True)
        log_prefix = os.path.join(args.log_dir, sanitize_name(run_name))
        lines = set_sbatch_option(lines, 'output', f'{log_prefix}-%j.out')
        lines = set_sbatch_option(lines, 'error', f'{log_prefix}-%j.err')

    header = '\n'.join(lines)
    repo_root = os.path.abspath(args.repo_root)
    pythonpath_line = f'\nexport PYTHONPATH="{repo_root}:${{PYTHONPATH:-}}"\n'
    run_string_cmd = f"\npython {args.launch} -config {config_path} --remove_scheduler --reset_step"
    return header + pythonpath_line + run_string_cmd, run_string_cmd

def parse_sbatch_job_id(stdout):
    match = re.search(r'Submitted batch job (\d+)', stdout)
    return match.group(1) if match else None

def prepare_job_ids_file(path):
    if not path:
        return
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w'):
        pass

def append_job_id(path, job_id, name, config_path, script_path):
    if not path:
        return
    with open(path, 'a') as f:
        f.write(f'{job_id}\t{name}\t{config_path}\t{script_path}\n')

def main(args):
    template = OmegaConf.load(args.template)
    copies = [OmegaConf.create({k:template[k].copy() if not isinstance(template[k], str) else template[k] for k in template['template_info']['include_keys']}) for i in range(template['template_info']['create'])]

    for i in range(len(copies)):
        for template_key in template['template_info']['template_keys']:
            val_to_set = attrgetter(template_key)(copies[i])[i]
            rsetattr(copies[i], template_key, val_to_set)

        if 'wandb' in copies[i]: 
            if 'update_config_with_wandb_id' not in copies[i]['wandb']: copies[i]['wandb']['update_config_with_wandb_id'] = True

    if args.deterministic_names:
        names = [sanitize_name(copies[i].get('wandb', {}).get('name', f'run_{i}')) for i in range(len(copies))]
    else:
        names = [f'{i}_{random.randint(0,1000000)}' for i in range(len(copies))]

    save_dir = args.save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not args.dry_run:
        prepare_job_ids_file(args.job_ids_out)

    launched_jobs = []
    for i in range(len(copies)):
        config_path = os.path.join(save_dir, f'{names[i]}.yaml')
        script_path = os.path.join(save_dir, f'{names[i]}.sh')
        OmegaConf.save(copies[i], config_path)
        run_string, run_string_cmd = build_run_string(args, config_path, names[i])
        with open(script_path, 'w') as f:
            f.write(run_string)
        if args.dry_run:
            print(f'Prepared {names[i]} - {run_string_cmd} - mode: {args.mode} - script: {script_path}')
            continue
        result = subprocess.run(['sbatch', script_path], check=True, capture_output=True, text=True)
        print(result.stdout.strip())
        job_id = parse_sbatch_job_id(result.stdout)
        if job_id is None:
            raise RuntimeError(f'Could not parse sbatch job id from output: {result.stdout}')
        launched_jobs.append((job_id, names[i], config_path, script_path))
        append_job_id(args.job_ids_out, job_id, names[i], config_path, script_path)
        print(f'Launched {names[i]} as {job_id} - {run_string_cmd} - mode: {args.mode}')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-template','--template', type=str, required=True, help='Path to the template config file')
    parser.add_argument('-mode','--mode', type=str, default='a100', help='denotes launch string to use to start slurm script')
    parser.add_argument('-l', '--launch', default='train.py', help='Path to the training script')
    parser.add_argument('--save_dir', default=SAVE_DIR, help='Directory for generated configs and Slurm scripts')
    parser.add_argument('--log_dir', default=None, help='Directory for Slurm stdout/stderr logs')
    parser.add_argument('--job_ids_out', default=None, help='Optional TSV path recording submitted job ids')
    parser.add_argument('--job_name_prefix', default=None, help='Optional prefix for Slurm job names')
    parser.add_argument('--repo_root', default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), help='Repo root to prepend to PYTHONPATH in generated Slurm scripts')
    parser.add_argument('--deterministic_names', action='store_true', help='Name generated configs/scripts from wandb.name instead of random ids')
    parser.add_argument('--dry_run', action='store_true', help='Generate configs/scripts without submitting sbatch jobs')
    parser.add_argument('--time', default=None, help='Override #SBATCH --time')
    parser.add_argument('--mem', default=None, help='Override #SBATCH --mem')
    parser.add_argument('--partition', default=None, help='Override #SBATCH --partition')
    parser.add_argument('--gres', default=None, help='Override #SBATCH --gres')
    parser.add_argument('--qos', default=None, help='Override #SBATCH --qos')
    parser.add_argument('--cpus_per_task', default=None, help='Override #SBATCH --cpus-per-task')
    args = parser.parse_args()
    main(args)
