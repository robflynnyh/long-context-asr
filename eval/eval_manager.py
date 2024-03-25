'''code for running evals over multiple datasets'''
import argparse
from omegaconf import OmegaConf
import os
import pandas as pd
from tqdm import tqdm

from run import main as run_eval, datasets_functions
accepted_splits = ['test', 'dev', 'train', 'all']

class ArgsClass():
    def __init__(self, args_dict):
        self.__dict__.update(args_dict)

    def __contains__(self, key):
        return key in self.__dict__.keys()

def checks(config):
    for dataset in config.datasets:
        assert dataset.name in datasets_functions.keys(), f'Dataset {dataset} not found! must be one of {datasets_functions.keys()}'
    for model in config.models:
        assert os.path.exists(model.path), f'Checkpoint {model.path} does not exist'
    assert os.path.exists("/".join(config.args.save_dataframe_path.split("/")[:-1])), f'dataframe save directory {"/".join(config.args.save_dataframe_path.split("/")[:-1])} does not exist'

def get_args(config, split, model, dataset_config):
    return ArgsClass({
        'checkpoint': model.path,
        'split': split,
        'seq_len': model.seq_len,
        'overlap': int(model.seq_len * model.get('overlap_ratio', 0.875)),
        'dataset': dataset_config.name,
        **model.get('args', {}),
        **config.get('args', {}),
        **dataset_config.get('args', {})
    })


def get_data_to_save(config, wer, split, dataset, model):
    data = {
        'dataset': dataset,
        'split': split,
        'wer': [wer],
        'name': model.name,
        'checkpoint': model.path,
        'repeat': model.repeat,
        'seq_len': model.seq_len,
        'overlap_ratio': [model.overlap_ratio],
        'model_class': config.args.model_class,
    }   
    return data

def check_if_already_evaluated(model_save_path, cur_df, dataset, split, args):
    '''
    ADD CHECKS FOR DATASET ASWELL AND SPLIT AS MODEL CAN BE EVALUATED ON MULTIPLE DATASETS AND SPLITS (currently only checks model_save_path)
    '''
    # check if a model with the same checkpoint path has already been evaluated
    if cur_df is None:
        return False
    
    cur_df = cur_df.loc[cur_df['checkpoint'] == model_save_path].loc[cur_df['dataset'] == dataset].loc[cur_df['split'] == split]
    cur_df = cur_df.loc[cur_df['seq_len'] == args.seq_len]

    model = cur_df
    if len(model) == 0:return False
    else: return True
       
def main(args, config):
    datasets = list(set([el.name for el in config.datasets]))
    checks(config)

    print(f'Running evals on datasets: {", ".join(datasets)}')
    print(f'Checkpoints to evaluate: {len(config.models)}')
    total_evals = len(config.models) * sum([len(config.datasets[ix].splits) for ix, el in enumerate(datasets)]) 
    print(f'Total number of evals: {total_evals}')

    cur_df = pd.read_csv(config.args.save_dataframe_path) if os.path.exists(config.args.save_dataframe_path) else None
 
    evals_completed = 0
    pbar = tqdm(total=total_evals, desc='Evaluations completed')
    results = []

    for dataset_config in config.datasets:
        dataset_name = dataset_config.name
        dataset_splits = dataset_config.splits
        dataset_reference = dataset_config.get('reference', dataset_name)
        for split in dataset_splits:
            for model in config.models:
                args = get_args(config, split, model, dataset_config)
                if check_if_already_evaluated(model.path, cur_df, dataset=dataset_reference, split=split, args=args): print(f'Skipping {model.path} as it has already been evaluated'); continue
                wer, model_config = run_eval(args = args)
                data_to_save = get_data_to_save(config, wer, split, dataset_reference, model)
                df = pd.DataFrame(data_to_save)
                df.to_csv(config.args.save_dataframe_path, mode='a', header=not os.path.exists(config.args.save_dataframe_path)) if config.args.save_dataframe_path != '' else None
                evals_completed += 1
                pbar.update(1)
                results.append(data_to_save)
  
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', '--config', type=str, default='eval_config.yaml', help='path to config file for eval')
    args = parser.parse_args()
    args.log = ''
    
    assert os.path.exists(args.config), f'Config file {args.config} does not exist'
    config = OmegaConf.load(args.config)
    main(args, config)