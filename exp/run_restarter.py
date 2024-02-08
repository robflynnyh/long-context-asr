import argparse
from omegaconf import OmegaConf
from os.path import join, exists
import time
from operator import attrgetter
import random
import subprocess
# set random seed based on time
random.seed(int(time.time()*10000))

run_strings = {
    'a100':f"""#!/bin/bash\n
#SBATCH --time=90:00:00
#SBATCH --mem=150GB
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate a100

""",
    'h100':f"""#!/bin/bash\n
#SBATCH --time=90:00:00
#SBATCH --mem=160GB
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:h100:1   
#SBATCH --qos=gpu
#SBATCH --cpus-per-task=8

module load Anaconda3/2022.10
source activate a100

"""
}


def main(args):
    assert args.mode in run_strings, f'Invalid mode {args.mode} selected. Valid modes are {list(run_strings.keys())}'
    assert exists(args.tmp_dir), f'No directory found at {args.tmp_dir}'

    for run_name in args.run_names:
        config_path, launch_path = join(args.tmp_dir, f'{run_name}.yaml'), join(args.tmp_dir, f'{run_name}.sh')
        
        if not exists(config_path):
            raise ValueError(f'No config file found at {config_path}')
        
        if not args.keep_seed:
            config = OmegaConf.load(config_path)
            config['training']['random_seed'] = random.randint(0, 1000000) # set new random seed for restart
            OmegaConf.save(config, config_path) # save new config

        run_string_cmd = f"\npython {args.launch} -config {config_path} -num_workers 0"
        run_string = run_strings[args.mode] + run_string_cmd
        with open(launch_path, 'w') as f:
            f.write(run_string)
        subprocess.run(['sbatch', launch_path])
        print(f'Restarted {run_name} - {run_string_cmd} - mode: {args.mode}')


        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-run_names','--run_names', required=True, nargs='+', help='denotes the names of the runs to be launched')
    parser.add_argument('-tmp_dir', '--tmp_dir', type=str, default='./.tmp', help='denotes the directory to save run configs and lauch scripts with names corresponding to run_names (created by run_launcher.py)')
    parser.add_argument('-mode','--mode', type=str, default='a100', help='denotes launch string to use to start slurm script')
    parser.add_argument('-l', '--launch', default='train.py', help='Path to the training script')
    parser.add_argument('-keep_seed', '--keep_seed', action='store_true', help='If true, will not re-randomize the seed for each restart (we do this by default to avoid dodgy batch that caused the crash)')
    args = parser.parse_args()
    main(args)
