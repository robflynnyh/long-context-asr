import run
import argparse
import os

def main(args):
    checkpoints = os.listdir(args.checkpoint_folder)
    for cpt in checkpoints:
        if cpt.endswith('.pt'):
            cpt_path = os.path.join(args.checkpoint_folder, cpt)
            print('Evaluating checkpoint:', cpt_path)
            args.checkpoint = cpt_path
            run.main(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-cf', '--checkpoint_folder', type=str, help='path to checkpoint folder', required=True)

    parser.add_argument('-split', '--split', type=str, default='test', help='test or dev split')
    parser.add_argument('-seq', '--seq_len', type=int, default=-1, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-overlap', '--overlap', type=int, default=0, help='-1 to use setting from config in checkpoint file')
    parser.add_argument('-beams', '--beam_width', type=int, default=1, help='beam width for decoding')
    parser.add_argument('-cache_len', '--cache_len', type=int, default=-1, help='cache length for decoding')

    parser.add_argument('-arpa', '--arpa_path', type=str, default='', help='path to arpa file')
    parser.add_argument('-alpha', '--alpha', type=float, default=0.5, help='alpha for lm')
    parser.add_argument('-beta', '--beta', type=float, default=0.8, help='beta for lm')

    parser.add_argument('-single_utt', '--single_utterance', action='store_true', help='single utterance decoding')
    parser.add_argument('-nv', '--not_verbose', action='store_true', help='verbose')

    parser.add_argument('-log', '--log', type=str, default='')

    args = parser.parse_args()
    args.verbose = not args.not_verbose

    main(args)


