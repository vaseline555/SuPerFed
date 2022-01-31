import os
import sys
import time
import pickle
import argparse
import threading
import logging

from torch.utils.tensorboard import SummaryWriter

from src.server import Server
from src.utils import launch_tensor_board

def main(args, writer):
    """Main program to run LG TMS Anomaly Detction.
    
    Args:
        args: user input arguments parsed by argparser
        writer: SummaryWriter instance for TensorBoard tracking
    
    Retunrs:
    """
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
        
    # initialize federated learning 
    central_server = Server(writer, args)
    central_server.setup()

    # do federated learning
    central_server.fit(args.experiment)

    # save resulting losses and metrics
    with open(os.path.join(log_path, f"{args.experiment}.pkl"), "wb") as f:
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
    
    # experiment related arguments
    parser.add_argument('--dataset', help='name of dataset to use for an experiment: [MNIST|CIFAR10|FEMNIST|TinyImageNet|Shakespeare|CIFAR100|EMNIST]', type=str, choices=['MNIST', 'CIFAR10', 'FEMNIST', 'ResNet18', 'TinyImageNet', 'Shakespeare', 'CIFAR100', 'EMNIST'])
    parser.add_argument('--num_shards', help='how many shards to be assigned for each client ignored if `iid=True` for pathological non-IID setting (MNIST & CIFAR10)', type=int)
    parser.add_argument('--iid', help='whether to simulate statistical homogeneity across clients (yes if NOT passed)', default='store_true')
    parser.add_argument('--a', help='shape parameter for a Dirichlet distribution used for splitting data in non-IID manner', type=float, default=0.5)
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
    parser.add_argument('--model_name', help='model to use [TwoNN|TwoCNN|NextCharLSTM|ResNet18|MobileNetv2]', type=str, choices=['TwoNN', 'TwoCNN', 'NextCharLSTM', 'ResNet18', 'MobileNetv2'])
    parser.add_argument('--init_type', type=str, help='initialization type [normal|xavier|xavier_uniform|kaiming|orthogonal|none]', default='xavier', choices=['xavier', 'normal', 'kaiming', 'xavier_uniform', 'orthogonal', 'none'])
    parser.add_argument('--init_gain', type=float, help='init gain for init type', default=1.0)
    parser.add_argument('--init_seed', help='init seeds for subspace learning (two different seeds need to be passed)', nargs='+', default=[])
    parser.add_argument('--fc_type', help='type of fully connected layer', type=str, choices=['StandardLinear', 'LinesLinear'], default='StandardLinear')
    parser.add_argument('--conv_type', help='type of fully connected layer', type=str, choices=['StandardConv', 'LinesConv'], defulat='StandardConv')
    parser.add_argument('--bn_type', help='type of fully batch normalization layer', type=str, choices=['StandardBN', 'LinesBN'])
    parser.add_argument('--embedding_type', help='type of embedding layer', type=str, choices=['StandardEmbedding', 'LinesEmbedding'])
    parser.add_argument('--lstm_type', help='type of LSTM layer', type=str, choices=['StandardLSTM', 'LinesLSTM'])
    parser.add_argument('--in_channels', help='input channels for image dataset', type=int, default=3)
    parser.add_argument('--num_classes', help='number of classes to predict', type=int)
    
    # parse arguments
    args = parser.parse_args()
    
    # modify log_path to contain current time
    log_path = f'{args.log_path}/{args.dataset}/{args.experiment}'

    # initiate TensorBaord for tracking losses and metrics
    writer = SummaryWriter(log_dir=log_path, filename_suffix=args.experiment)
    tb_thread = threading.Thread(
        target=launch_tensor_board,
        args=([log_path, args.tb_port, args.tb_host])
        ).start()
    time.sleep(3.0)

    # set global logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        filename=os.path.join(log_path, f"{args.experiment}.log"),
        level=logging.INFO,
        format="[%(levelname)s](%(asctime)s) %(message)s",
        datefmt="%Y/%m/%d/ %I:%M:%S %p"
    )
    
    # display and log experiment configuration
    message = "\n[WELCOME] Unfolding configurations...!"
    print(message); logging.info(message)
    for arg in vars(args):
        print(f'[INFO] {arg}: {getattr(args, arg)}')
        logging.info(arg); logging.info(getattr(args, arg))
    print()
    
    # run main program!
    main(args, writer)
    
    # bye!
    message = "[INFO] ...done all learning process!\n[INFO] ...exit program!"
    print(message); logging.info(message)
    time.sleep(3); sys.exit(0)

