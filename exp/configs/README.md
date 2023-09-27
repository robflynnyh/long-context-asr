# Instructions for creating config files
Below is an example config file with all fields explained

```
model:
  feat_in: 80 # input feature dimension (number of mel bins)
  n_layers: 6 # number of layers
  d_model: 768 # main model dimension
  n_heads: 6 # number of heads
  head_dim: 128 # dimension of each head
  dropout_ff: 0.0 # dropout in feedforward layers
  dropout_attn: 0.0 # dropout in attention layers
  dropout_conv: 0.0 # dropout in convolution layers
  subsampling_factor: 8 # subsampling factor i.e 8x subsampling
  subsampling: dw_striding # subsampling method either dw_striding or striding - dw denotes the use of depthwise separable convolutions
  subsampling_act: relu # activation function for subsampling layers can be relu/silu/gelu
  subsampling_conv_channels: 256 # number of channels in subsampling convolution layers -1 defaults to d_model
  expansion_factor: 4 # expansion factor for feedforward modules
  self_condition_subsampling: false # whether to use self conditioning on the output of subsampling layers
  subsampling_norm_out: false # whether to use layer norm on the output of subsampling layers
  conv_kernel_size: 9 # kernel size for convolution layers in conformer blocks
  conv_expansion_factor: 1 # expansion factor for convolution layers in conformer blocks
  conv_type: 'standard' # can be standard or longconv. standard = conformer paper, longconv = long convolutions (https://arxiv.org/abs/2302.06646)
  self_conditioning: true # whether to use self conditioning on the output of conformer blocks
  gated_sc: false # whether to gate the self conditioning output with a sigmoid instead of simply adding it back in as standard (not recommended)
  decoder_norm: true # whether to use layer norm on the output of conformer blocks
  use_rotary: true # whether to use rotary embeddings in attention layers
  default_norm: layer_norm # normalisation to use can either be layer_norm or rms_norm
  sandwich_norm: false # use normalisation method given in https://arxiv.org/abs/2105.13290 (not recommended) default is pre_norm
  bias_in_ff: false # whether to use bias in feedforward layers
  checkpoint_every_n_layers: 0 # whether to checkpoint every n layers (0 means no checkpointing) (gradient checkpointing)
  flash_attn: true # whether to use flash attention to compute attention (reccomended!!)


optimizer:
  name: madgrad # optimizer type either adam/madgrad/mirrormadgrad
  args: # these args are passed to the optimizer i.e. lr, weight_decay etc
    lr: 2.5e-3

scheduler: # we use a cosine decay scheduler
  warmup_steps: 4000 # denostes the number of warmup steps before the learning rate starts to decay

audio_chunking:
  size: 512 # size of audio chunks used during training 512 frames corresponds to 5.12 seconds

sequence_scheduler: # used to increase the sequence length during training (omit this section if you do not want to use this scheduler)
  increase_every: 5000 # increase the sequence length every 5000 steps
  stop_after: 90000 # stop increasing the sequence length after 90000 steps
  start_after: 0 # start the step counter at 0
  max_sequence_length: 16384 # don't increase the sequence length beyond 16384 frames
  increase_by_multiplier: 2 # increase the sequence length by a factor of 2 every time
  batch_size_multiplier: 0.5 # decrease the batch size by a factor of 2 every time

wandb:
  use: true # whether to use wandb to log training progress (recommended)
  project_name: spotify_long_context # name of the wandb project
  name: 'multi_epoch_run_16k' # name of the run
  id: '' # id of the run (use for resuming training of existing runs)

checkpointing:
  dir: /mnt/parscratch/users/acp21rjf/spotify/checkpoints/multi_epoch_run_16k # directory to save checkpoints to
  save_every_n_steps: 2000 # save a checkpoint every 2000 steps

data:
  path: /mnt/parscratch/users/acp21rjf/spotify/audio_txt_pairs.json # path to the json file containing the audio/text pairs creating using the preprocessing script

spec_augment:
  n_time_masks: 2 # Number of time masks. If its value is zero, no time masking will be applied
  time_mask_param: 100 # Maximum possible length of the time mask
  n_freq_masks: 2 # Number of frequency masks. If its value is zero, no frequency masking will be applied
  freq_mask_param: 27 # Maximum possible length of the frequency mask
  iid_masks: true # Applies iid masks to each of the examples in the batch dimension.
  p: 0.5 # maximum proportion of time steps that can be masked. Must be within range [0.0, 1.0]. (Default: 1.0)
  zero_masking: false # If true, masked values will be set to zero, otherwise they will be set to the mean value of the input spectrogram

training:
  start_spec_augment_after_n_epochs: 1 # -1 to disable (number of epochs to wait before starting spec augment)
  max_epochs: 9 # maximum number of epochs to train for
  batch_size: 704 # initial batch size
  backprop_every: 1 # backpropagate every n steps
  backwards_every: 1 # backwards pass every n steps
  clip_value: 0.8 # gradient norm clipping value

```
