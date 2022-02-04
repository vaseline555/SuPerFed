# pathological non-IID setting
## MNIST
python3 main.py --exp_name fedavg_mnist_patho_50 --tb_port 11047 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.2 --K 50 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 600 \
--eval_every 100

python3 main.py --exp_name fedavg_mnist_patho_100 --tb_port 11048 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 100

python3 main.py --exp_name fedavg_mnist_patho_500 --tb_port 11049 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.02 --K 500 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 60 \
--eval_every 100



## CIFAR10
python3 main.py --exp_name fedavg_cifar10_patho_50 --tb_port 31880 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.2 --K 50 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 500 \
--eval_every 100

python3 main.py --exp_name fedavg_cifar10_patho_100 --tb_port 31881 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 250 \
--eval_every 100

python3 main.py --exp_name fedavg_cifar10_patho_500 --tb_port 31882 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.02 --K 500 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 50 \
--eval_every 100