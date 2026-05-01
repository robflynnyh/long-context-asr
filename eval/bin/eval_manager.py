'''code for running evals over multiple datasets'''
import argparse
from lcasr.utils.omegaconf import OmegaConf
import os
import pandas as pd
from tqdm import tqdm
from lcasr.utils.helpers import ArgsClass
from run import main as run_eval, datasets_functions
accepted_splits = ['test', 'dev', 'train', 'all']


def get_model_metadata(model):
    metadata = {}
    for key, value in model.items():
        if key in {'args', 'path'}:
            continue
        metadata[key] = value
    return metadata


    

def checks(config):
    for dataset in config.datasets:
        assert dataset.name in datasets_functions.keys(), f'Dataset {dataset} not found! must be one of {datasets_functions.keys()}'
    for model in config.models:
        assert os.path.exists(model.path), f'Checkpoint {model.path} does not exist'
    assert os.path.exists("/".join(config.args.save_dataframe_path.split("/")[:-1])), f'dataframe save directory {"/".join(config.args.save_dataframe_path.split("/")[:-1])} does not exist'

def get_args(config, split, model, dataset_config):
    args = {
        'checkpoint': model.path,
        'split': split,
        'seq_len': model.seq_len,
        'overlap': int(model.seq_len * model.get('overlap_ratio', 0.875)),
        'dataset': dataset_config.name,
        **model.get('args', {}),
        **config.get('args', {}),
        **dataset_config.get('args', {})
    }
    # add evaryign from model dict that is not in args
    for key, value in model.items():
        if key not in args and key != 'args':
            args[key] = value
    return ArgsClass(args)


def get_data_to_save(config, wers, split, dataset, model):
    model_metadata = get_model_metadata(model)
    data = [{
        'dataset': dataset,
        'split': split,
        'wer': wer_data['wer'],
        'recording': wer_data['recording'],
        'words': wer_data['words'],
        'ins_rate': wer_data['ins_rate'],
        'del_rate': wer_data['del_rate'],
        'sub_rate': wer_data['sub_rate'],
        'name': model.name,
        'checkpoint': model.path,
        'repeat': model.repeat,
        'seq_len': model.seq_len,
        'overlap_ratio': model.overlap_ratio,
        'model_class': config.args.model_class,
    } for wer_data in wers]

    for row in data:
        for key, value in model_metadata.items():
            if key not in row:
                row[key] = value

    return data

def check_if_already_evaluated(model, cur_df, dataset, split):
    if cur_df is None:
        return False
    
    cur_df = cur_df.loc[cur_df['checkpoint'] == model.path].loc[cur_df['dataset'] == dataset].loc[cur_df['split'] == split]
    for key, value in get_model_metadata(model).items():
        if key in cur_df.columns:
            cur_df = cur_df.loc[cur_df[key] == value]

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
                if check_if_already_evaluated(model, cur_df, dataset=dataset_reference, split=split): print(f'Skipping {model.path} as it has already been evaluated'); continue
                wers, model_config = run_eval(args = args)
                data_to_save = get_data_to_save(config, wers, split, dataset_reference, model)
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
