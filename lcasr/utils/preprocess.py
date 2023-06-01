from lcasr.utils import audio_tools
import argparse
import os
from os.path import join
from tqdm import tqdm
import torch

def del_spectograms(ogg_path):
    '''
    delete all spectograms in all folders
    '''
    ogg_dir = ogg_path
    for top_level in [f'{el}' for el in range(8)]:
        print(f'processing {top_level}')
        path = join(ogg_dir, top_level)
        folders = os.listdir(path)
        for folder in tqdm(folders, desc='folders'):
            path = join(ogg_dir, top_level, folder)
            shows = os.listdir(path)
            for show in tqdm(shows, desc='shows'):
                path = join(ogg_dir, top_level, folder, show)
                episodes = os.listdir(path)
                for ogg in episodes:
                    path = join(ogg_dir, top_level, folder, show, ogg)
                    if path.endswith('.spec.pt'):
                        os.remove(path)

def stage_1(args):
    '''
    In this stage we convert ogg files to spectograms and save them in the same folder with same name except with file extension .spec.pt
    '''
    ogg_dir = args.ogg_path
    shows = os.listdir(ogg_dir)
    for show in tqdm(shows, desc='shows'):
        path = join(ogg_dir, show)
        episodes = os.listdir(path)
        for ogg in episodes:
            path = join(ogg_dir, show, ogg)
            if path.endswith('.ogg'):
                out_path = path.replace('.ogg', '.spec.pt')
                if not os.path.exists(out_path):
                    spectogram = audio_tools.processing_chain(path).to(torch.float16)
                    torch.save(spectogram, out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ogg_path', type=str, default='')
    parser.add_argument('--stage', type=str, default='0', help='0: convert ogg to spectograms')

    args = parser.parse_args()

    stage = int(args.stage)
    
    if stage == 0:
        assert os.path.exists(args.ogg_path), 'ogg_path does not exist'
        stage_1(args)
