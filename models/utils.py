# %% Modules

import sys
import argparse

from IPython.core.debugger import Pdb

# %% Functions

def ipdb_breakpoint(*args, **kwargs):
    Pdb(context=21).set_trace(sys._getframe().f_back)

sys.breakpointhook = ipdb_breakpoint

def build_parser():
    parser = argparse.ArgumentParser()
    subparser = parser.add_subparsers(dest="command", required=True)

    train_parser = subparser.add_parser("train")
    train_parser.add_argument("--input", type=str, required=True)
    train_parser.add_argument("--checkpoint", type=str, required=True)
    train_parser.add_argument("--block-size", type=int, default=128)
    train_parser.add_argument("--batch-size", type=int, default=32)
    train_parser.add_argument("--d-model", type=int, default=128)
    train_parser.add_argument("--num-heads", type=int, default=4)
    train_parser.add_argument("--num-layers", type=int, default=4)
    train_parser.add_argument("--learning-rate", type=float, default=3.0e-04)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--grad-clip", type=float, default=1.0)
    train_parser.add_argument("--epochs", type=int, default=5)
    train_parser.add_argument("--eval-every", type=int, default=200)
    train_parser.add_argument("--prompt", type=str, default="Making sure this works. This is")
    train_parser.add_argument("--max-new-tokens", type=int, default=300)
    train_parser.add_argument("--temperature", type=float, default=0.8)

    evaluate_parser = subparser.add_parser("evaluate")
    evaluate_parser.add_argument("--prompt", type=str, required=True)
    evaluate_parser.add_argument("--checkpoint", type=str, required=True)

    return parser

# %% End of script
