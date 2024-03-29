# pathological non-IID setting
#########
# MNIST #
#########
# FedAvg
python3 main.py --exp_name fedavg_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 600 \
--eval_every 50 &
python3 main.py --exp_name fedavg_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 50 &
python3 main.py --exp_name fedavg_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 60 \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 600 \
--eval_every 50 &
python3 main.py --exp_name fedprox_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 50 &
python3 main.py --exp_name fedprox_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 60 \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm scaffold --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 600 \
--eval_every 50 &
python3 main.py --exp_name scaffold_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm scaffold --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 50 &
python3 main.py --exp_name scaffold_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm scaffold --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 60 \
--eval_every 50 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 600 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 300 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm lg-fedavg  --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 60 --n_jobs 10 \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedper --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 600 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedper_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 300 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedper_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedper --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 60 --n_jobs 10 \
--eval_every 50 &

# APFL
python3 main.py --exp_name apfl_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 600 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name apfl_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 300 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name apfl_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 60 --n_jobs 10 \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --lr 0.01 --mu 15 \
--split_type pathological --shard_size 600 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name pfedme_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --lr 0.01 --mu 15 \
--split_type pathological --shard_size 300 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name pfedme_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --lr 0.01 --mu 15 \
--split_type pathological --shard_size 60 --n_jobs 10 \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedrep --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --tau 5 \
--split_type pathological --shard_size 600 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedrep_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --tau 5 \
--split_type pathological --shard_size 300 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedrep_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedrep --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --tau 5 \
--split_type pathological --shard_size 60 --n_jobs 10 \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --tau 5 --mu 1 \
--split_type pathological --shard_size 600 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name ditto_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --tau 5 --mu 1 \
--split_type pathological --shard_size 300 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name ditto_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --tau 5 --mu 1 \
--split_type pathological --shard_size 60 --n_jobs 10 \
--eval_every 50 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --L 0.6 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 600 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --L 0.4 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 300 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --L 0.2 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 60 \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --L 0.8 --nu 2 --mu 0.01 \
--split_type pathological --shard_size 600 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_mnist_patho_100 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --L 0.6 --nu 2 --mu 0.01 \
--split_type pathological --shard_size 300 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_mnist_patho_500 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --L 0.4 --nu 2 --mu 0.01 \
--split_type pathological --shard_size 60 \
--eval_every 50 &


###########
# CIFAR10 #
###########
# FedAvg
python3 main.py --exp_name fedavg_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedavg_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedavg_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedprox_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedprox_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm scaffold --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name scaffold_cifar10_patho_100 --tb_port 6662 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm scaffold --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name scaffold_cifar10_patho_500 --tb_port 6663 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm scaffold --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_cifar10_patho_100  \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedper --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedper_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedper_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedper --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# APFL
python3 main.py --exp_name apfl_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name apfl_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name apfl_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 10 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --mu 15 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name pfedme_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --mu 15 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name pfedme_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --mu 15  \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedrep --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --tau 5 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedrep_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --tau 5 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name fedrep_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedrep --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --tau 5 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --tau 5 --mu 1 \
--split_type pathological --shard_size 500 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name ditto_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --tau 5 --mu 1 \
--split_type pathological --shard_size 250 --n_jobs 10 \
--eval_every 50 &
python3 main.py --exp_name ditto_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --tau 5 --mu 1 \
--split_type pathological --shard_size 50 --n_jobs 10 \
--eval_every 50 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --L 0.6 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 500 \
--eval_every 50 & 
python3 main.py --exp_name superfed-mm_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --L 0.4 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 250 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --L 0.2 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 50 \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar10_patho_50 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --L 0.8 --nu 2 --mu 0.01 \
--split_type pathological --shard_size 500 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_cifar10_patho_100 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 10 --L 0.6 --nu 2 --mu 0.01 \
--split_type pathological --shard_size 250 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_cifar10_patho_500 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 10 --L 0.4 --nu 2 --mu 0.01 \
--split_type pathological --shard_size 50 \
--eval_every 50 &
