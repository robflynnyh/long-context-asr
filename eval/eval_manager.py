'''code for running evals over multiple datasets'''
import argparse
from omegaconf import OmegaConf
import os
import pandas as pd
from tqdm import tqdm
from earnings22.run import main as run_earnings22
from tedlium.run import main as run_tedlium
from rev16.run import main as run_rev16
from tedlium_concat.run import main as run_tedlium_concat

dataset_funcs = {
    'earnings22': run_earnings22,
    'tedlium': run_tedlium,
    'tedlium_concat': run_tedlium_concat, 
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
        'verbose': False,
    })

def get_data_to_save(config, wer, split, dataset, model):
    data = {
        'dataset': dataset,
        'split': split,
        'wer': [wer],
        'name': model.name,
        'checkpoint': model.path,
        'repeat': model.repeat,
        'single_utterance': config.args.single_utterance if dataset in singlue_utterance_datasets else False,
        'seq_len': model.seq_len,
        'overlap_ratio': [model.overlap_ratio],
        'model_class': config.args.model_class,
        'cache_len': -1,
    }   
    return data

def check_if_already_evaluated(model_save_path, cur_df, dataset, split):
    '''
    ADD CHECKS FOR DATASET ASWELL AND SPLIT AS MODEL CAN BE EVALUATED ON MULTIPLE DATASETS AND SPLITS (currently only checks model_save_path)
    '''
    # check if a model with the same checkpoint path has already been evaluated
    if cur_df is None:
        return False
    
    model = cur_df.loc[cur_df['checkpoint'] == model_save_path].loc[cur_df['dataset'] == dataset].loc[cur_df['split'] == split]
    if len(model) == 0:return False
    else: return True
       
def main(args, config):
    datasets = config.args.datasets
    checks(config)

    print(f'Running evals on datasets: {", ".join(datasets)}')
    print(f'Checkpoints to evaluate: {len(config.models)}')
    print(f'Evaluating on splits: {", ".join(config.args.splits)}')
    total_evals = len(datasets) * len(config.models) * len(config.args.splits)
    print(f'Total number of evals: {total_evals}')

    cur_df = pd.read_csv(config.args.save_dataframe_path) if os.path.exists(config.args.save_dataframe_path) else None

    evals_completed = 0
    pbar = tqdm(total=total_evals, desc='Evaluations completed')
    results = []
    for dataset in datasets:
        for split in config.args.splits:
            if dataset == 'rev16' and split == 'dev': continue # rev16 does not have a dev split
            for model in config.models:
                if check_if_already_evaluated(model.path, cur_df, dataset=dataset, split=split): print(f'Skipping {model.path} as it has already been evaluated'); continue

                args = get_args(config, split, model)
                wer, model_config = dataset_funcs[dataset](args)
                data_to_save = get_data_to_save(config, wer, split, dataset, model)
                results.append(data_to_save)
                df = pd.DataFrame(data_to_save)
                df.to_csv(config.args.save_dataframe_path, mode='a', header=not os.path.exists(config.args.save_dataframe_path)) if config.args.save_dataframe_path != '' else None
                evals_completed += 1
                pbar.update(1)
    
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, default='eval_config.yaml', help='path to config file for eval')
    args = parser.parse_args()
    args.log = ''
    
    assert os.path.exists(args.config), f'Config file {args.config} does not exist'
    config = OmegaConf.load(args.config)
    main(args, config)