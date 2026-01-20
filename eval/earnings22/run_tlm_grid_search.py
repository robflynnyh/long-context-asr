import argparse
from tlm_beam import main as run_tlm_search
from lcasr.utils.general import argsclass


def create_args_from_setup(args, decoding_setup, cache_length, logits_path, max_recordings=-1) -> argsclass:
    new_args = argsclass(
        beam_width = 25,
        alpha = decoding_setup['alpha'],
        beta = decoding_setup['beta'],
        p = decoding_setup['p'],
        use_gpu = args.use_gpu,
        no_ray = True,
        checkpoint = args.checkpoint,
        logits_path = logits_path,
        use_wandb = False,
        max_len=cache_length,
        use_init_cache = (not args.no_init_cache),
        max_recordings = max_recordings,
        debug_with_greedy = False,
        log_path = ""
    )
    return new_args

def main(args):
    cache_lengths = [64, 128, 256, 512, 1024, 2048]
    
    decoding_setups = [
        {
            'alpha': 0.42,
            'beta': 1.95,
            'p': 2.96,
        },
        {
            'alpha': 0.35,
            'beta': 2.19,
            'p': 3.03,
        },
        {
            'alpha': 0.53,
            'beta': 2.97,
            'p': 3.53,
        },
        {
            'alpha': 0.55,
            'beta': 2.92,
            'p': 2.89,
        },
        {
            'alpha': 0.55,
            'beta': 3.0,
            'p': 3.23,
        },
        {
            'alpha': 0.54,
            'beta': 2.96,
            'p': 3.245,
        },
        {
            'alpha': 0.53,
            'beta': 3.12,
            'p': 3.51,
        }
    ]


    for i in range(len(cache_lengths)):
        # wers = []
        # for j in range(len(decoding_setups)):
        #     args_setup = create_args_from_setup(
        #         args,
        #         decoding_setup=decoding_setups[j],
        #         cache_length=cache_lengths[i],
        #         logits_path=args.logits_path_dev,
        #         max_recordings=2,
        #     )
        #     wer_dev = run_tlm_search(args_setup)
        #     wers.append(wer_dev)
        #     print(f"Dev WER for cache length {cache_lengths[i]}, setup {j}: {wer_dev}")
        # best_setup_index = wers.index(min(wers))
        best_setup_index = 0
        # print(f"Best setup for cache length {cache_lengths[i]}: {decoding_setups[best_setup_index]} with WER {wers[best_setup_index]}")
        args_setup_test = create_args_from_setup(
            args,
            decoding_setup=decoding_setups[best_setup_index],
            cache_length=cache_lengths[i],
            logits_path=args.logits_path_test,
        )
        wer_test = run_tlm_search(args_setup_test)
        print(f"Test WER for cache length {cache_lengths[i]}: {wer_test}")

        if args.log_path != "":
            with open(args.log_path, 'a') as f:
                f.write(f"Cache Length: {cache_lengths[i]}, alpha: {decoding_setups[best_setup_index]['alpha']}, beta: {decoding_setups[best_setup_index]['beta']}, p: {decoding_setups[best_setup_index]['p']}, Test WER: {wer_test}\n")






if __name__ == '__main__':
    parser = argparse.ArgumentParser("Running grid search for TLM beam search decoding")
    parser.add_argument('--logits_path_dev', type=str, required=True, help='path to logits for dev set')
    parser.add_argument('--logits_path_test', type=str, required=True, help='path to logits for test set')
    parser.add_argument('--checkpoint', type=str, help='path to llm checkpoint', default='/mnt/parscratch/users/acp21rjf/language_modelling/spotipile/512_1280/step_540012.pt') 
    parser.add_argument('--use_gpu', action='store_true', help='use gpu for decoding')
    parser.add_argument('--log_path', type=str, default='', help='path to log file')
    parser.add_argument('--no_init_cache', action='store_true', help='do not use initial cache')
   
    args = parser.parse_args()
    main(args)