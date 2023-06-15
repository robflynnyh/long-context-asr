import os
import subprocess

TOPDIRS = [f'{el}' for el in range(8)]
BASEPATH = '/mnt/parscratch/users/acp21rjf/spotify/audio/'

def create_bash_file(path):
    '''
    create bash file to run on cluster
    '''
    save_name = path.replace('/', '_') + '.sh'
    with open(save_name, 'w') as f:
        f.write(\
f'''#!/bin/bash
#SBATCH --time=03:30:00
#SBATCH --mem=32GB

module load Anaconda3/2022.10
source activate a100
python -m lcasr.utils.preprocess --ogg_path {path} --stage 0
        ''')
    return save_name
        
for topdir in TOPDIRS:
    path = os.path.join(BASEPATH, topdir)
    folders = os.listdir(path)
    for folder in folders:
        path = os.path.join(BASEPATH, topdir, folder)
        bfile = create_bash_file(path)
        # lauch job
        subprocess.run(['sbatch', bfile])




