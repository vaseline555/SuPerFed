import os
import time
import datetime
import pickle
import argparse
import threading
import logging

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'experiment',
        help='name of experiment',
        type=str
    )
    parser.add_argument(
        "--global_seed",
        help='global random seed (applied EXCEPT model initiailization)',
        type=int,
        default=5959
    )
    parser.add_argument(
        '--device',
        help='device to use, either cpu or cuda; default is cpu',
        type=str,
        default="cpu",
        choices=["cpu", "cuda"]
    )
    parser.add_argument(
        '--data_path',
        help='path where data is stored',
        default='./data'
    )
    parser.add_argument(
        '--dataset',
        help='name of dataset name to do an experiment (MNIST, CIFAR10, CIFAR100, EMNIST, FEMNIST, TinyImageNet)',
        type=str
    )
    parser.add_argument(
        '--num_shards',
        help='how many shards to be assigned for each client ignored if `iid=True`',
        type=int
    )
    parser.add_argument(
        '--iid',
        help='split manner of dataset (0: False, 1: True)',
        type=int,
        default=0,
        choices=[0, 1]
    )
    parser.add_argument(
        '--C',
        help='sampling fraction of clietns per each round',
        type=float,
        default=0.1
    )
    parser.add_argument(
        '--K',
        help='number of total cilents',
        type=int,
        default=100
    )
    parser.add_argument(
        '--R',
        help='number of total federation round',
        type=int,
        default=100
    )
    parser.add_argument(
        '--E',
        help='number of local epochs',
        type=int,
        default=10
    )
    parser.add_argument(
        '--B',
        help='batch size for local update in each client',
        type=int,
        default=10
    )
    parser.add_argument(
        '--L',
        help='when to start local training round - float: start local model training from (L * R) round',
        type=float,
        default=0.8
    )
    parser.add_argument(
        '--lr',
        help='learning rate of each client',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--mu',
        help='proximity regularization term',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--beta',
        help='subspace construction term',
        type=float,
        default=1.0
    )
    parser.add_argument(
        '--p_lr',
        help='learning rate of personalization round per each client',
        type=float,
        default=0.01
    )
    parser.add_argument(
        '--p_e',
        help='number of personalization update per each client',
        type=int,
        default=1
    )
    parser.add_argument(
        "--init_type",
        type=str,
        help='initialization type `(xavier, normal, kaiming)`',
        default='xavier',
        choices=['xavier', 'normal', 'kaiming']
    )
    parser.add_argument(
        "--init_gain",
        type=float,
        help='init gain for init type',
        default=1.0
    )
    parser.add_argument(
        "--init_seed",
        help='init seeds for subspace learning',
        nargs='+',
        default=[]
    )
    parser.add_argument(
        "--model_name",
        help='model config: name of a model `(MNISTConvNet, CIFARConvNet, TINConvNet)`',
        type=str,
        choices=['MNISTConvNet', 'CIFARConvNet', 'TINConvNet']
    )
    parser.add_argument(
        "--in_channels",
        help='model config: input channels',
        type=int
    )
    parser.add_argument(
        "--hidden_channels",
        help='model config: hidden channels',
        type=int,
        default=32
    )
    parser.add_argument(
        "--num_hiddens",
        help='model config: number of hidden nodes in a hidden layer',
        type=int,
        default=512
    )
    parser.add_argument(
        "--num_classes",
        help='model config: number of classes (output)',
        type=int
    )
    parser.add_argument(
        "--log_path",
        help='path to store logs',
        type=str,
        default='./log'
    )
    parser.add_argument(
        "--tb_port",
        help='TensorBoard port number',
        type=int
    )
    parser.add_argument(
        "--tb_host",
        help='TensorBoard host address',
        type=str,
        default='0.0.0.0'
    )
    
    # parse arguments
    args = parser.parse_args()
    
    # change pre-defined argument
    if args.dataset == "EMNIST":
        args.K = 1543
    elif args.dataset == "TinyImageNet":
        args.K = 388
    
    # check if correct model is specified
    if args.dataset in ["MNINST", "EMNIST"]:
        assert args.model_name == 'MNISTConvNet'
    elif args.dataset in ["CIFAR10", "CIFAR100"]:
        assert args.model_name == 'CIFARConvNet'
    elif args.dataset in ["TinyImageNet"]:
        assert args.model_name == 'TINConvNet'
    
    # modify log_path to contain current time
    #log_path = os.path.join(f'{args.log_path}/{args.dataset}/{args.experiment}', str(datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")))
    log_path = f'{args.log_path}/{args.dataset}/{args.experiment}'

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_path, filename_suffix=args.experiment)
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_path, args.tb_port, args.tb_host])
        ).start()
    time.sleep(3.0)

    # set the configuration of global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_path, f"{args.experiment}.log"),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p")
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)
    
    for arg in vars(args):
        print(f'[INFO] {arg}: {getattr(args, arg)}')
        logging.info(arg); logging.info(getattr(args, arg))
    print()
    
    # initialize federated learning 
    central_server = Server(writer, args)
    central_server.setup()

    # do federated learning
    central_server.fit(args.experiment)

    # save resulting losses and metrics
    with open(os.path.join(log_path, f"{args.experiment}.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)
    
    # bye!
    message = "...done all learning process!\n...exit program!"
    print(message); logging.info(message)
    time.sleep(3); exit(0)

