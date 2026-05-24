import torch, argparse, lcasr
import importlib
from lcasr.eval.utils import fetch_logits as moving_average_eval
from lcasr.eval.buffered_transcription import fetch_logits as buffered_eval
from lcasr.utils.general import load_model, get_model_class
from lcasr.eval.wer import word_error_rate_detail 
#from lcasr.eval.dynamic_eval import dynamic_eval
from lcasr.decoding.greedy import GreedyCTCDecoder
from whisper.normalizers import EnglishTextNormalizer
normalize = EnglishTextNormalizer()
from tqdm import tqdm

DATASET_MODULES = {
    'earnings22_full': ('earnings22_full.run', 'get_text_and_audio'),
    'earnings22': ('earnings22.run', 'get_text_and_audio'),
    'tedlium': ('tedlium.run', 'get_text_and_audio'),
    'rev16': ('rev16.run', 'get_text_and_audio'),
    'this_american_life': ('this_american_life.run', 'get_text_and_audio'),
    'spotify': ('spotify.run', 'get_text_and_audio'),
    'floras50': ('floras50.run', 'get_text_and_audio'),
}


datasets_functions = {name: None for name in DATASET_MODULES}


def get_dataset_function(dataset):
    module_name, function_name = DATASET_MODULES[dataset]
    module = importlib.import_module(module_name)
    return getattr(module, function_name)


def get_transcribe_kwargs(args, verbose):
    transcribe_kwargs = args.__dict__.get('transcribe_kwargs', {})
    if transcribe_kwargs is None:
        transcribe_kwargs = {}
    if not hasattr(transcribe_kwargs, 'items'):
        raise TypeError('transcribe_kwargs must be a mapping')

    transcribe_kwargs = {key: value for key, value in transcribe_kwargs.items()}
    transcribe_kwargs.setdefault('verbose', verbose)
    return transcribe_kwargs


def length_diagnostics(hypotheses, references):
    return {
        'hyp_words': sum(len(text.split()) for text in hypotheses),
        'ref_words': sum(len(text.split()) for text in references),
        'hyp_chars': sum(len(text.replace(" ", "")) for text in hypotheses),
        'ref_chars': sum(len(text.replace(" ", "")) for text in references),
    }


def main(args):
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_config = checkpoint['config']
    args.config = model_config

    if args.__dict__.get('disable_flash_attention', False): args.config.model.flash_attn = False

    eval_fn = moving_average_eval
    if args.__dict__.get('evaluation_mode', 'averaged_moving_window') == 'windowed_attention':
        if args.__dict__.get('window_size', None) is None:
            seq_len = args.seq_len
            subsample_factor = args.config.model.get('subsampling_factor', 8)
            ds_seq_len = seq_len // subsample_factor
            window_size = ds_seq_len // 2 # //2 because applied in both directions
        else: window_size = args.window_size
        args.config.model.attention_window_size = window_size
        args.seq_len = args.__dict__.get('max_sequence_length', 3600000) # 10 hours
    if args.__dict__.get('evaluation_mode', 'averaged_moving_window') == 'buffered': eval_fn = buffered_eval
    if args.__dict__.get('overide_window_size', None) is not None:
        args.config.model.attention_window_size = args.overide_window_size
    
    include_per_recording_evaluations = args.__dict__.get('include_per_recording_evaluations', False)

    verbose = args.__dict__.get('verbose', True)   
    transcribe_kwargs = get_transcribe_kwargs(args, verbose)

    tokenizer = {}
    tokenizer_path = args.__dict__.get("tokenizer_path", None)
    if tokenizer_path is not None:
        args.tokenizer_path = tokenizer_path
        tokenizer = {"tokenizer_path": args.tokenizer_path}
        print("Using tokenizer path from args:", args.tokenizer_path)

    tokenizer = lcasr.utils.audio_tools.load_tokenizer(**tokenizer)
    model = load_model(args.config, tokenizer.vocab_size(), model_class=get_model_class({'model_class': args.config.get('model_class', args.model_class)}))
    model.print_total_params()
    model.load_state_dict(checkpoint['model'], strict=False)
    print(f'Loaded model from {args.checkpoint}')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.device = device
    model = model.to(device)
    model.eval()

    if not hasattr(model, 'transcribe'): decoder = GreedyCTCDecoder(tokenizer = tokenizer, blank_id = model.decoder.num_classes-1)

    data = get_dataset_function(args.dataset)(args.split)

    # for idx, module in enumerate([el.attend.fn for el in model.layers]):
    #     module.return_attention_weights = True

    all_texts = []
    all_golds = []
    wer_data = []

    pbar = tqdm(range(len(data)), total=len(data)) #if verbose else range(len(data))
    for rec in pbar:
        if verbose: print(f'Processing {rec+1}/{len(data)}')

        if verbose: print('\n-------\n'+data[rec]['id']+'\n-------\n')
        
        audio_spec, gold_text = data[rec]['process_fn'](data[rec])
        

        if hasattr(model, 'transcribe'):
            all_text = model.transcribe(
                audio_spec,
                tokenizer,
                device=device,
                max_sequence_length=args.seq_len,
                **transcribe_kwargs,
            )
            out = normalize(all_text).lower().strip()
            
        else: # assume ctc
            logits = eval_fn(
                args = args, 
                model = model, 
                spec = audio_spec,
                seq_len = args.seq_len,
                overlap = args.overlap,
                tokenizer = tokenizer
            ) 
            out_text = decoder(torch.as_tensor(logits))
            out = normalize(out_text).lower()
        
        if verbose: print(gold_text, '\n', out, '\n\n')
        
        all_texts.append(out)
        all_golds.append(gold_text)

        # wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
        # print(wer)
        # exit()

        if include_per_recording_evaluations:
            wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text])
            cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(hypotheses=[out], references=[gold_text], use_cer=True)
            wer_data.append({
                'recording': data[rec]['id'],
                'wer': wer,
                'cer': cer,
                'words': words,
                'chars': chars,
                'ins_rate': ins_rate,
                'del_rate': del_rate,
                'sub_rate': sub_rate,
                'char_ins_rate': char_ins_rate,
                'char_del_rate': char_del_rate,
                'char_sub_rate': char_sub_rate,
                **length_diagnostics([out], [gold_text]),
            })

        if args.__dict__.get('break_eval', False): break

        

    wer, words, ins_rate, del_rate, sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds)
    cer, chars, char_ins_rate, char_del_rate, char_sub_rate = word_error_rate_detail(hypotheses=all_texts, references=all_golds, use_cer=True)

    if verbose: print(f'WER: {wer}')

    wer_data.append({
        'recording': 'all',
        'wer': wer,
        'cer': cer,
        'words': words,
        'chars': chars,
        'ins_rate': ins_rate,
        'del_rate': del_rate,
        'sub_rate': sub_rate,
        'char_ins_rate': char_ins_rate,
        'char_del_rate': char_del_rate,
        'char_sub_rate': char_sub_rate,
        **length_diagnostics(all_texts, all_golds),
    })
    return wer_data, model_config
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', '-d', type=str, default='earnings22', choices=DATASET_MODULES.keys())

    parser.add_argument('-c', '--checkpoint', type=str, default='../../exp/model.pt', help='path to checkpoint')
    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-model_class', '--model_class', type=str, default='SCConformerXL', help='model class')
    parser.add_argument('-repeat', '--repeat', type=int, default=1, help='number of times to rerun evaluation')
    parser.add_argument('-eval_mode', '--evaluation_mode', type=str, default='averaged_moving_window', choices=['averaged_moving_window', 'windowed_attention', 'buffered'])

    parser.add_argument('-break', '--break_eval', action='store_true', help='break after first recording') 
    args = parser.parse_args()
    main(args)
    

#python run.py -d earnings22 -r 3 -dfa -epochs 5 -kwargs optim_lr=0.00009 spec_augment_freq_mask_param=34 spec_augment_min_p=0.1879883950862319 spec_augment_n_time_masks=0 spec_augment_n_freq_masks=6

#CUDA_VISIBLE_DEVICES="1" python run.py -dfa -epochs 5 -seq 16384 -o 14336 -split test --dataset earnings22 -r 3 -s "./results/earnings22.json" -kwargs optim_lr=9e-5 spec_augment_freq_mask_param=34 spec_augment_min_p=0.18 spec_augment_n_freq_masks=6  spec_augment_n_time_masks=0 
