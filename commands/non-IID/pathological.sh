# pathological non-IID setting
#########
# MNIST #
#########
# FedAvg
python3 main.py --exp_name fedavg_mnist_patho_50 --tb_port 8619 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name fedavg_mnist_patho_100 --tb_port 8620 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name fedavg_mnist_patho_500 --tb_port 8621 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# FedProx
python3 main.py --exp_name fedprox_mnist_patho_50 --tb_port 8625 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name fedprox_mnist_patho_100 --tb_port 8626 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name fedprox_mnist_patho_500 --tb_port 8627 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.01 --K 500 --R 200 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# APFL
python3 main.py --exp_name apfl_mnist_patho_50 --tb_port 6658 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name apfl_mnist_patho_100 --tb_port 6659 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name apfl_mnist_patho_500 --tb_port 6660 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# pFedMe
python3 main.py --exp_name pfedme_mnist_patho_50 --tb_port 6664 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 20 --mu 15 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name pfedme_mnist_patho_100 --tb_port 6665 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 --mu 15 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name pfedme_mnist_patho_500 --tb_port 6666 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 20 --mu 15 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_mnist_patho_50 --tb_port 21897 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name lg-fedavg_mnist_patho_100 --tb_port 21898 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name lg-fedavg_mnist_patho_500 --tb_port 21899 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm lg-fedavg  --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# FedPer
python3 main.py --exp_name fedper_mnist_patho_50 --tb_port 21903 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedper --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name fedper_mnist_patho_100 --tb_port 21904 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name fedper_mnist_patho_500 --tb_port 21905 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedper --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# FedRep
python3 main.py --exp_name fedrep_mnist_patho_50 --tb_port 23341 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedrep --model_name TwoNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name fedrep_mnist_patho_100 --tb_port 23342 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name fedrep_mnist_patho_500 --tb_port 23343 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm fedrep --model_name TwoNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# Ditto
python3 main.py --exp_name ditto_mnist_patho_50 --tb_port 23347 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name ditto_mnist_patho_100 --tb_port 23348 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name ditto_mnist_patho_500 --tb_port 23349 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_mnist_patho_50 --tb_port 28931 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 20 --L 0.2 --nu 5 --mu 0 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name superfed-mm_mnist_patho_100 --tb_port 28932 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 --L 0.2 --nu 5 --mu 0 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name superfed-mm_mnist_patho_500 --tb_port 28933  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 10 --B 20 --L 0.2 --nu 5 --mu 0 \
--split_type pathological --shard_size 60 \
--eval_every 500 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_mnist_patho_50 --tb_port 28937 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 5 --B 10 --L 0.2 --nu 1 --mu 0 \
--split_type pathological --shard_size 600 \
--eval_every 500 &
python3 main.py --exp_name superfed-lm_mnist_patho_100 --tb_port 28938 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 5 --B 10 --L 0.2 --nu 1 --mu 0 \
--split_type pathological --shard_size 300 \
--eval_every 500 &
python3 main.py --exp_name superfed-lm_mnist_patho_500 --tb_port  28939 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 500 --E 5 --B 10 --L 0.2 --nu 1 --mu 0 \
--split_type pathological --shard_size 60 \
--eval_every 500 &


###########
# CIFAR10 #
###########
# FedAvg
python3 main.py --exp_name fedavg_cifar10_patho_50 --tb_port 8622 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name fedavg_cifar10_patho_100 --tb_port 8623 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name fedavg_cifar10_patho_500 --tb_port 8624 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# FedProx
python3 main.py --exp_name fedprox_cifar10_patho_50 --tb_port 8628 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name fedprox_cifar10_patho_100 --tb_port 8629 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name fedprox_cifar10_patho_500 --tb_port 8630 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# APFL
python3 main.py --exp_name apfl_cifar10_patho_50 --tb_port 6661 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name apfl_cifar10_patho_100 --tb_port 6662 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name apfl_cifar10_patho_500 --tb_port 6663 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# pFedMe
python3 main.py --exp_name pfedme_cifar10_patho_50 --tb_port 6667 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 20 --mu 15 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name pfedme_cifar10_patho_100 --tb_port 6668 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 --mu 15 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name pfedme_cifar10_patho_500 --tb_port 6669 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 20 --mu 15  \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_cifar10_patho_50 --tb_port 21900 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name lg-fedavg_cifar10_patho_100 --tb_port 21901 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name lg-fedavg_cifar10_patho_500 --tb_port 21902 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# FedPer
python3 main.py --exp_name fedper_cifar10_patho_50 --tb_port 21906 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedper --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name fedper_cifar10_patho_100 --tb_port 21907 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name fedper_cifar10_patho_500 --tb_port 21908 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedper --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# FedRep
python3 main.py --exp_name fedrep_cifar10_patho_50 --tb_port 23344 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedrep --model_name TwoCNN \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name fedrep_cifar10_patho_100 --tb_port 23345 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name fedrep_cifar10_patho_500 --tb_port 23346 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedrep --model_name TwoCNN \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# Ditto
python3 main.py --exp_name ditto_cifar10_patho_50 --tb_port 23350 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name ditto_cifar10_patho_100 --tb_port 23351 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name ditto_cifar10_patho_500 --tb_port 23352 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 20 \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar10_patho_50 --tb_port 28934 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 20 --L 0.2 --nu 5 --mu 0  \
--split_type pathological --shard_size 500 \
--eval_every 500 & 
python3 main.py --exp_name superfed-mm_cifar10_patho_100 --tb_port 28935 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 --L 0.2 --nu 5 --mu 0  \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name superfed-mm_cifar10_patho_500 --tb_port 28936  \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 20 --L 0.2 --nu 5 --mu 0  \
--split_type pathological --shard_size 50 \
--eval_every 500 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar10_patho_50 --tb_port 28940 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.1 --K 50 --R 500 --E 10 --B 20 --L 0.2 --nu 1 --mu 0  \
--split_type pathological --shard_size 500 \
--eval_every 500 &
python3 main.py --exp_name superfed-lm_cifar10_patho_100 --tb_port 28941 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 --L 0.2 --nu 1 --mu 0  \
--split_type pathological --shard_size 250 \
--eval_every 500 &
python3 main.py --exp_name superfed-lm_cifar10_patho_500 --tb_port 28941 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.01 --K 500 --R 500 --E 10 --B 20 --L 0.2 --nu 1 --mu 0  \
--split_type pathological --shard_size 50 \
--eval_every 500 &