import os
import sys
import json
import time
import torch
import random
import argparse
import threading
import numpy as np

from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board, initiate_model, get_dataset
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
    # get dataset
    split_map, server_testset, client_datasets = get_dataset(args)
    
    
    
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

    
    
    ##############
    # Run server #
    ##############
    # create central server
    central_server = Server(args, writer, model, server_testset, client_datasets)
    
    # initialize central server
    central_server.setup()

    # do federated learning
    central_server.fit()

    # save resulting losses and metrics
    with open(os.path.join(args.result_path, f'{args.exp_name}/final_result.json'), 'w') as result_file:
        arguments = {'arguments': {str(arg): getattr(args, arg) for arg in vars(args)}}
        sample_stats = {'sample_statistics': split_map}
        results = {'results': {key: value for key, value in central_server.results.items() if len(value) > 0}}
        json.dump({**arguments, **sample_stats, **results}, result_file)

if __name__ == "__main__":
    # parse user inputs as arguments
    parser = argparse.ArgumentParser()
    
    # default arguments
    parser.add_argument('--exp_name', help='experiment name', type=str, required=True)
    parser.add_argument('--global_seed', help='global random seed (applied EXCEPT model initiailization)', type=int, default=5959)
    parser.add_argument('--device', help='device to use, either cpu or cuda; default is cpu', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--data_path', help='data path', type=str, default='./data')
    parser.add_argument('--log_path', help='log path', type=str, default='./log')
    parser.add_argument('--result_path', help='result path', type=str, default='./result')
    parser.add_argument('--plot_path', help='plot path', type=str, default='./plot')
    parser.add_argument('--tb_port', help='TensorBoard port number', type=int, default=6006)
    parser.add_argument('--tb_host', help='TensorBoard host address', type=str, default='0.0.0.0')
    
    # dataset related arguments
    parser.add_argument('--dataset', help='name of dataset to use for an experiment: [MNIST|CIFAR10|CIFAR100|TinyImageNet|FEMNIST|Shakespeare|]', type=str, choices=['MNIST', 'CIFAR10', 'CIFAR100', 'TinyImageNet', 'FEMNIST', 'Shakespeare'], required=True)
    parser.add_argument('--is_small', help='indicates the size of inputs is small; only used for MobileNetv2 (if passed)', action='store_true')
    parser.add_argument('--in_channels', help='input channels for image dataset (ignored when `Shakespeare` dataset is used)', type=int, default=3)
    parser.add_argument('--num_classes', help='number of classes to predict (ignored when `Shakespeare` dataset is used)', type=int, default=10)
    parser.add_argument('--test_fraction', help='fraction of test dataset at each client', type=float, default=0.2)
    
    # label noise experiment
    parser.add_argument('--label_noise', help='experiment under the simulation of a label noise (if passed)', action='store_true')
    parser.add_argument('--noise_type', help='type of a label noise: [pair|symmetric]', type=str, choices=['pair', 'symmetric'])
    parser.add_argument('--noise_rate', help='label noise ratio (0 ~ 1) valid only when `label-noise` argument is passed', type=float, default=0.2)
    
    # dataset split scenario
    parser.add_argument('--split_type', help='type of an expriment to conduct', type=str, choices=['iid', 'pathological', 'dirichlet', 'realistic'], required=True)
    parser.add_argument('--shard_size', help='size of one shard to be assigned to each client; used only when `algo_type=pathological`', type=int, default=300)
    parser.add_argument('--alpha', help='shape parameter for a Dirichlet distribution used for splitting data in non-IID manner; used only when `algo_type=dirichlet`', type=float, default=0.5)
    
    # federated learning arguments
    parser.add_argument('--algorithm', help='type of an algorithm to use', type=str, choices=['fedavg', 'fedprox', 'lg-fedavg', 'fedper', 'fedrep', 'ditto', 'apfl', 'pfedme', 'superfed-mm', 'superfed-lm'], required=True)
    parser.add_argument('--C', help='sampling fraction of clietns per each round', type=float, default=0.1)
    parser.add_argument('--K', help='number of total cilents', type=int, default=100)
    parser.add_argument('--R', help='number of total federated learning rounds', type=int, default=1000)
    parser.add_argument('--E', help='number of local epochs', type=int, default=10)
    parser.add_argument('--B', help='batch size for local update in each client', type=int, default=10)
    parser.add_argument('--L', help='when to start local training round (start local model training from `floor(L * R)` round)', type=float, default=0.8)
    parser.add_argument('--eval_every', help='evaluate at every `eval_every` round', type=int, default=100)
    
    # optimization related arguments
    parser.add_argument('--lr', help='learning rate of each client', type=float, default=0.01)
    parser.add_argument('--lr_decay', help='magnitude of learning rate decay at every round', type=float, default=0.99)
    parser.add_argument('--mu',help='constant for proximity regularization term (for fedprox, ditto, pfedme, superfed)', type=float, default=0.01)
    parser.add_argument('--tau', help='constant for fine tuning head or updating a local model (for fedrep, ditto, pfedme)', type=int, default=5)
    parser.add_argument('--alpha', help='constant for mixing models (for apfl)', type=float, default=0.25)
    
    # model related arguments
    parser.add_argument('--model_name', help='model to use [TwoNN|TwoCNN|NextCharLM|ResNet18|MobileNetv2]', type=str, choices=['TwoNN', 'TwoCNN', 'NextCharLM', 'ResNet18', 'MobileNetv2'])
    parser.add_argument('--init_type', type=str, help='initialization type [normal|xavier|xavier_uniform|kaiming|orthogonal|none]', default='xavier', choices=['xavier', 'normal', 'kaiming', 'xavier_uniform', 'orthogonal', 'none'])
    parser.add_argument('--init_gain', type=float, help='init gain for init type', default=1.0)
    parser.add_argument('--fc_type', help='type of fully connected layer', type=str, choices=['StandardLinear', 'LinesLinear'], default='StandardLinear')
    parser.add_argument('--conv_type', help='type of fully connected layer', type=str, choices=['StandardConv', 'LinesConv'], default='StandardConv')
    parser.add_argument('--bn_type', help='type of fully batch normalization layer', type=str, choices=['StandardBN', 'LinesBN'], default='StandardBN')
    parser.add_argument('--embedding_type', help='type of embedding layer', type=str, choices=['StandardEmbedding', 'LinesEmbedding'], default='StandardEmbedding')
    parser.add_argument('--lstm_type', help='type of LSTM layer', type=str, choices=['StandardLSTM', 'LinesLSTM'], default='StandardLSTM')
    
    # parse arguments
    args = parser.parse_args()
        
    # make path for saving losses & metrics
    if not os.path.exists(os.path.join(args.result_path, args.exp_name)):
        os.makedirs(os.path.join(args.result_path, args.exp_name))
    
    # make path for saving plots
    if not os.path.exists(os.path.join(args.plot_path, args.exp_name)):
        os.makedirs(os.path.join(args.plot_path, args.exp_name))
        
    # define path to save a log
    args.log_path = f'{args.log_path}/{args.exp_name}'

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=args.log_path, filename_suffix=str(args.global_seed))
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([args.log_path, args.tb_port, args.tb_host])
        ).start()
    time.sleep(3.0)
    
    # display and log experiment configuration
    print('\n[WELCOME] Configurations...')
    for arg in vars(args):
        print(f'\t * {str(arg).upper()}: {getattr(args, arg)}')
    
    # run main program
    main(args, writer)
    
    # bye!
    print('[INFO] ...done federated learning!\n[INFO] ...exit program!')
    sys.exit(0)
