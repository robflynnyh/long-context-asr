import audio_tools
import argparse
import os
from os.path import join
from tqdm import tqdm
import torch

def stage_1(args):
    '''
    In this stage we convert ogg files to spectograms and save them in the same folder with same name except with file extension .spec.pt
    '''
    ogg_dir = args.ogg_path
    top_level_split = args.top_level_split

    path = join(ogg_dir, top_level_split)
    folders = os.listdir(path)
    for folder in tqdm(folders, desc='folders'):
        path = join(ogg_dir, top_level_split, folder)
        shows = os.listdir(path)
        for show in tqdm(shows, desc='shows'):
            path = join(ogg_dir, top_level_split, folder, show)
            episodes = os.listdir(path)
            for ogg in episodes:
                path = join(ogg_dir, top_level_split, folder, show, ogg)
                if path.endswith('.ogg'):
                    out_path = path.replace('.ogg', '.spec.pt')
                    if not os.path.exists(out_path):
                        spectogram = audio_tools.processing_chain(path)
                        torch.save(spectogram, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ogg_path', type=str)
    parser.add_argument('--top_level_split', type=str, default='0', help='corpus is partitioned into folders 0-7, this dictates which folder to process, this is done to parallelize the process')
    parser.add_argument('--stage', type=str, default='0', help='0: convert ogg to spectograms')

    args = parser.parse_args()

    stage = int(args.stage)
    
    if stage == 0:
        stage_1(args)
