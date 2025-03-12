# coding=UTF-8
import sys
import os
sys.path.append(os.getcwd())
import importlib
import numpy as np

_cpath_='/usr/local/python/3.7.7/lib/python3.7/site-packages'
if _cpath_ in sys.path:
    sys.path.remove(_cpath_)
from mydatasets import NAMES as DATASET_NAMES
sys.path.insert(50, _cpath_)

from models import get_all_models
from argparse import ArgumentParser
from utils.args import add_management_args
from mydatasets import ContinualDataset
from utils.continual_training import train as ctrain
from mydatasets import get_dataset
from models import get_model
from utils.training import train
from utils.best_args import best_args
from utils.conf import set_random_seed
import torch
import torchvision
import torch.distributed as dist
from confusion_matrix import plot_confusion
import random
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = ArgumentParser(description='mammoth', allow_abbrev=False)
    parser.add_argument('--model', type=str, required=True,
                        help='Model name.', choices=get_all_models())
    parser.add_argument('--load_best_args', action='store_true',
                        help='Loads the best arguments for each method, '
                             'dataset and memory buffer.')
    parser.add_argument("--local_rank", type=int, default=0, help='node rank for distributed training')
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--num_classes", default=36, type=int)
    #! for random ER
    parser.add_argument("--rand_balance_cls", action='store_true', help='use class balance random or not ')

    #! for attack
    parser.add_argument("--use_inf", action='store_true', help='use inf or not ')
    parser.add_argument("--use_attack", action='store_true', help='use attack or not ')
    parser.add_argument("--use_ema", action='store_true', help='use ema or not ')
    parser.add_argument("--use_perturbation", action='store_true', help='use perturbation or not ')
    parser.add_argument("--num_good", type=float, default=None,\
                        help="num of good imgs, if none, only store good.",)
    parser.add_argument("--add_adv", action='store_true', help="add adv imgs or original imgs.",)
    parser.add_argument("--steps", type=int, default=100,help="the steps for attack",)
    parser.add_argument("--eps", type=float, default=16,help="the eps for attack",)
    parser.add_argument("--alpha_atk", type=float, default=2,help="the alpha for attack",)
    parser.add_argument("--use_l2", action='store_true', help='use l2 distance or not ',)
    parser.add_argument("--buffer_attack", type=str, default=None,\
                        help="the attack type, None means no attack in buffer",)
    parser.add_argument("--lamda_grad_norm", type=float, default=0.1,\
                        help="the lamda of grad_norm in grad_loss function",)
    parser.add_argument("--save_img", action='store_true', help='save imgs or not',)
    

    #! to store output results
    parser.add_argument("--out_dir", type=str, default=r'/home/bqqi/CVT/CL_Transformer-main/results/test',\
                         help="the folder to store output imgs",)
    

    add_management_args(parser)
    args = parser.parse_known_args()[0]
    mod = importlib.import_module('models.' + args.model)

    if args.load_best_args:
        parser.add_argument('--dataset', type=str, required=True,
                            choices=DATASET_NAMES,
                            help='Which dataset to perform experiments on.')
        if hasattr(mod, 'Buffer'):
            parser.add_argument('--buffer_size', type=int, required=True,
                                help='The size of the memory buffer.')
        args = parser.parse_args()
        if args.model == 'joint':
            best = best_args[args.dataset]['sgd']
        else:
            best = best_args[args.dataset][args.model]
        if args.model == 'joint' and args.dataset == 'mnist-360':
            args.model = 'joint_gcl'
        if hasattr(args, 'buffer_size'):
            best = best[args.buffer_size]
        else:
            best = best[-1]
        for key, value in best.items():
            setattr(args, key, value)
    else:
        get_parser = getattr(mod, 'get_parser')
        parser = get_parser()
        args = parser.parse_args()

    if args.seed is not None:
        set_random_seed(args.seed)  

    setattr(args, 'GAN', 'GAN')
    setattr(args, 'use_albumentations', False)
    setattr(args, 'use_apex', False)
    setattr(args, 'use_distributed', True)
    setattr(args, 'use_lr_scheduler', False)
    if torch.cuda.device_count() <= 1 or args.dataset == 'seq-mnist':
        setattr(args, 'use_distributed', False)

    if args.model == 'mer': setattr(args, 'batch_size', 1)
    dataset = get_dataset(args)
    if args.model == 'our' or args.model == 'our_reservoir':
        backbone = dataset.get_backbone_our()
    elif args.model == 'onlinevt':
        backbone = dataset.get_backbone_cct()
    else:
        backbone = dataset.get_backbone()

    loss = dataset.get_loss()
    model = get_model(args, backbone, loss, dataset.get_transform())
    model.to(model.device)

    print(args)
    if hasattr(model, 'loss_name'):
        print('loss name: ', model.loss_name)
        
    train(model, dataset, args)


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    # setup_seed(42)
    main()





