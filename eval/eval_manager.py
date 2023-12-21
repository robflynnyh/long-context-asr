'''code for running evals over multiple datasets'''
import argparse
from omegaconf import OmegaConf
import os
import pandas as pd
from tqdm import tqdm
from earnings22.run import main as run_earnings22
from tedlium.run import main as run_tedlium
from rev16.run import main as run_rev16

dataset_funcs = {
    'earnings22': run_earnings22,
    'tedlium': run_tedlium,
    'rev16': run_rev16
}
singlue_utterance_datasets = 'tedlium'
accepted_splits = ['test', 'dev']

class ArgsClass():
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)

    def __contains__(self, key):
        return key in self.__dict__.keys()

def checks(config):
    for dataset in config.args.datasets:
        assert dataset in dataset_funcs.keys(), f'Dataset {dataset} not found! must be one of {dataset_funcs.keys()}'
    for split in config.args.splits:
        assert split in accepted_splits, f'Split {split} not found! must be one of {accepted_splits}'  
    for model in config.models:
        assert os.path.exists(model.path), f'Checkpoint {model.path} does not exist'
    assert os.path.exists("/".join(config.args.save_dataframe_path.split("/")[:-1])), f'dataframe save directory {"/".join(config.args.save_dataframe_path.split("/")[:-1])} does not exist'

def get_args(config, split, model):
    return ArgsClass({
        'log': '',
        'checkpoint': model.path,
        'split': split,
        'seq_len': model.seq_len,
        'overlap': int(model.seq_len * model.overlap_ratio),
        'model_class': config.args.model_class,
        'cache_len': -1,
        'single_utterance': config.args.single_utterance,
    })

def get_data_to_save(wer, split, dataset, model, model_config):
    return {
        'dataset': dataset,
        'split': split,
        'wer': wer,
        'model': model_config,
        'name': model.name,
        'checkpoint': model.path,
        'repeat': model.repeat,
        'single_utterance': config.args.single_utterance if dataset in singlue_utterance_datasets else False,
        'seq_len': model.seq_len,
        'overlap': int(model.seq_len * model.overlap_ratio),
        'model_class': config.args.model_class,
        'cache_len': -1,
    }   

def main(args, config):
    datasets = config.args.datasets
    checks(config)

    print(f'Running evals on datasets: {", ".join(datasets)}')
    print(f'Checkpoints to evaluate: {len(config.models)}')
    print(f'Evaluating on splits: {", ".join(config.args.splits)}')
    total_evals = len(datasets) * len(config.models) * len(config.args.splits)
    print(f'Total number of evals: {total_evals}')

    evals_completed = 0
    pbar = tqdm(total=total_evals, desc='Evaluations completed')
    for dataset in datasets:
        for split in config.args.splits:
            for model in config.models:
                args = get_args(config, split, model)
                wer, model_config = dataset_funcs[dataset](args)
                data_to_save = get_data_to_save(wer, split, dataset, model, model_config)
                df = pd.DataFrame(data_to_save)
                df.to_csv(config.args.save_dataframe_path, mode='a', header=not os.path.exists(config.args.save_dataframe_path))
                evals_completed += 1
                pbar.update(1)
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, default='eval_config.yaml', help='path to config file for eval')
    args = parser.parse_args()
    args.log = ''
    
    assert os.path.exists(args.config), f'Config file {args.config} does not exist'
    config = OmegaConf.load(args.config)
    main(args, config)