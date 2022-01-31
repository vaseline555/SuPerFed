import os
import sys
import time
import torch
import random
import pickle
import argparse
import threading
import logging
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board, initiate_model
from src.models.builder import Builder
from src.models import models

def main(args, writer):
    """Main program to run LG TMS Anomaly Detction.
    
    Args:
        args: user input arguments parsed by argparser
        writer: SummaryWriter instance for TensorBoard tracking
    """
    ########
    # Seed #
    ########
    # set seed for reproducibility
    torch.manual_seed(args.global_seed)
    random.seed(args.global_seed)
    np.random.seed(args.global_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    
    ###################
    # Prepare dataset #
    ###################
    # split dataset
    
    
    
    
    #################
    # Prepare model #
    #################
    # check if correct model is specified
    builder = Builder(args)
    if 'ResNet' in args.model_name:
        block = models.BasicBlock
    elif 'MobileNet' in args.model_name:
        block = models.InvertedBlock
    else:
        block = None
    model = getattr(models, args.model_name)(builder, args, block)
    model = initiate_model(model, args)
    
    
    
    ##############
    # Run server #
    ##############
    # initialize federated learning 
    central_server = Server(args, writer)
    central_server.setup()

    # do federated learning
    central_server.fit(args.exp_name)

    # save resulting losses and metrics
    with open(os.path.join(args.log_path, f"{args.exp_name}.pkl"), "wb") as f:
        pickle.dump(central_server.results, f)


if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser()
    
    # default arguments
    parser.add_argument('--exp_name', type=str, help='name of the experiment', required=True)
    parser.add_argument('--global_seed', help='global random seed (applied EXCEPT model initiailization)', type=int, default=5959)
    parser.add_argument('--device', help='device to use, either cpu or cuda; default is cpu', type=str, default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--data_path', help='data path', type=str, default='./data')
    parser.add_argument('--log_path', help='log path', type=str, default='./log')
    parser.add_argument('--tb_port', help='TensorBoard port number', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address', type=str, default='0.0.0.0')
    
    # dataset related arguments
    parser.add_argument('--dataset', help='name of dataset to use for an experiment: [MNIST|CIFAR10|FEMNIST|TinyImageNet|Shakespeare|CIFAR100|EMNIST]', type=str, choices=['MNIST', 'CIFAR10', 'FEMNIST', 'TinyImageNet', 'Shakespeare', 'CIFAR100', 'EMNIST'])
    parser.add_argument('--is_small', help='indicates the size of inputs is small (if passed)', default='store_true')
    parser.add_argument('--in_channels', help='input channels for image dataset (ignored when `Shakespeare` dataset is used)', type=int, default=3)
    parser.add_argument('--num_classes', help='number of classes to predict (ignored when `Shakespeare` dataset is used)', type=int, default=10)
    parser.add_argument('--num_shards', help='how many shards to be assigned for each client ignored if `iid=True` for pathological non-IID setting (MNIST & CIFAR10)', type=int)
    parser.add_argument('--iid', help='whether to simulate statistical homogeneity across clients (yes if passed)', default='store_true')
    parser.add_argument('--a', help='shape parameter for a Dirichlet distribution used for splitting data in non-IID manner', type=float, default=0.5)
    
    # federated learning arguments
    parser.add_argument('--C', help='sampling fraction of clietns per each round', type=float, default=0.1)
    parser.add_argument('--K', help='number of total cilents', type=int, default=100)
    parser.add_argument('--R', help='number of total federated learning rounds', type=int, default=1000)
    parser.add_argument('--E', help='number of local epochs', type=int, default=10)
    parser.add_argument('--B', help='batch size for local update in each client', type=int, default=10)
    parser.add_argument('--L', help='when to start local training round (start local model training from `floor(L * R)` round)', type=float, default=0.8)
    
    # optimization related arguments
    parser.add_argument('--lr', help='learning rate of each client', type=float, default=0.01)
    parser.add_argument('--mu',help='constant for proximity regularization term', type=float, default=1.0)
    parser.add_argument('--nu', help='constant for low-loss subspace construction term', type=float, default=1.0)
    parser.add_argument('--p_lr', help='learning rate of personalization round per each client', type=float, default=0.01)
    parser.add_argument('--p_e', help='number of personalization update per each client', type=int, default=1)
    
    # model related arguments
    parser.add_argument('--model_name', help='model to use [TwoNN|TwoCNN|NextCharLSTM|ResNet18|MobileNetv2]', type=str, choices=['TwoNN', 'TwoCNN', 'NextCharLM', 'ResNet18', 'MobileNetv2'])
    parser.add_argument('--init_type', type=str, help='initialization type [normal|xavier|xavier_uniform|kaiming|orthogonal|none]', default='xavier', choices=['xavier', 'normal', 'kaiming', 'xavier_uniform', 'orthogonal', 'none'])
    parser.add_argument('--init_gain', type=float, help='init gain for init type', default=1.0)
    parser.add_argument('--init_seed', help='init seeds for subspace learning (two different seeds need to be passed)', nargs='+', default=[5959])
    parser.add_argument('--fc_type', help='type of fully connected layer', type=str, choices=['StandardLinear', 'LinesLinear'], default='StandardLinear')
    parser.add_argument('--conv_type', help='type of fully connected layer', type=str, choices=['StandardConv', 'LinesConv'], default='StandardConv')
    parser.add_argument('--bn_type', help='type of fully batch normalization layer', type=str, choices=['StandardBN', 'LinesBN'], default='StandardBN')
    parser.add_argument('--embedding_type', help='type of embedding layer', type=str, choices=['StandardEmbedding', 'LinesEmbedding'], default='StandardEmbedding')
    parser.add_argument('--lstm_type', help='type of LSTM layer', type=str, choices=['StandardLSTM', 'LinesLSTM'], default='StandardLSTM')
    
    
    # parse arguments
    args = parser.parse_args()
    
    # check if arguments are specified correctly
    if 'Lines' in args.fc_type:
        assert len(args.init_seed) <= 2, '[ERROR] number of `init_seed` should be less than or equal to 2!'
    
    # define path to save a log
    args.log_path = f'{args.log_path}/{args.dataset}/{args.exp_name}'

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=args.log_path, filename_suffix=args.exp_name)
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([args.log_path, args.tb_port, args.tb_host])
        ).start()
    time.sleep(3.0)

    # set global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(args.log_path, f"{args.exp_name}.log"),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p"
    )
    
    # display and log experiment configuration
    message = "\n[WELCOME] Configurations..."
    print(message); logging.info(message)
    for arg in vars(args):
        print(f'\t *{arg}: {getattr(args, arg)}')
        logging.info(arg); logging.info(getattr(args, arg))
    print()
    
    # run main program
    main(args, writer)
    
    # bye!
    message = "[INFO] ...done all learning process!\n[INFO] ...exit program!"
    print(message); logging.info(message)
    time.sleep(3); sys.exit(0)

