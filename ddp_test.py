import argparse
import os

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
from transformers.deepspeed import HfDeepSpeedConfig
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
import deepspeed


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--master_addr', default='tcp://127.0.0.1', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--master_port', default='23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--ds_config', default='ds_config.json', type=str,
                    help='deepspeed config')

def main():
    args = parser.parse_args()
    dschf = HfDeepSpeedConfig(args.ds_config)
    args.dschf = dschf

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print(f'multi process WORLD SIZE {args.world_size} ngpus_per_node {ngpus_per_node}')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    global best_acc1
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            os.environ['RANK'] = str(args.rank)
            args.local_rank = args.rank * ngpus_per_node + gpu
            os.environ['WORLD_SIZE'] = str(args.world_size)
            os.environ['MASTER_ADDR'] = str(args.master_addr)
            os.environ['MASTER_PORT'] = str(args.master_port)
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            deepspeed.init_distributed(init_method=args.dist_url)
        # dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
        #                         world_size=args.world_size, rank=args.rank)

    # create model


    # For multiprocessing distributed, DistributedDataParallel constructor
    # should always set the single device scope, otherwise,
    # DistributedDataParallel will use all available devices.
    # model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')

    # if args.gpu is not None:
    #     torch.cuda.set_device(args.gpu)
    #     model.cuda(args.gpu)
    #     # When using a single GPU per process and per
    #     # DistributedDataParallel, we need to divide the batch size
    #     # ourselves based on the total number of GPUs we have
    #     args.batch_size = int(args.batch_size / ngpus_per_node)
    #     args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
    #     # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
    #     engine = deepspeed.initialize(model=model, config_params=args.ds_config)
    # else:
    #     model.cuda()
    #     # DistributedDataParallel will divide and allocate batch_size to all
    #     # available GPUs if device_ids are not set
    #     # model = torch.nn.parallel.DistributedDataParallel(model)
    #     engine = deepspeed.initialize(model=model, config_params=args.ds_config)
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased')
    model.cuda()
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr)
    model, optimizer, _, lr_scheduler = deepspeed.initialize(model=model, optimizer=optimizer, config_params=args.ds_config)
    if args.rank == 0:
        print('THIS IS MAIN')
        print(model)



if __name__ == '__main__':
    main()