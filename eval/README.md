# Code for Evaluating Models on selected datasets
- working datasets:
    - Earnings22 (https://arxiv.org/abs/2203.15591)
    - Rev16 (see whisper paper)
    - Tedlium (https://arxiv.org/abs/1805.04699)
    - This American Life (https://arxiv.org/abs/2005.08072)

- Run evaluations using the run.py file
- For running evaluations over multiple models and datasets see eval_manager.py examples of config files that can be used with this script canb be found in the eval_configs_for_journal folder
- There are 3 evaluation modes available which are: 'averaged_moving_window', 'windowed_attention', 'buffered'. averaged_moving_window is the setting used in the Interspeech 2024 paper.


