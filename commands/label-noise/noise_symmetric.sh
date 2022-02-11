# Label noise - Symmetric
## MNIST
# FedAvg
python3 main.py --exp_name fedavg_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedavg_mnist_patho_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# FedProx
python3 main.py --exp_name fedprox_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedprox_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name lg-fedavg_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# FedPer
python3 main.py --exp_name fedper_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedper_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# APFL
python3 main.py --exp_name apfl_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name apfl_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# pFedMe
python3 main.py --exp_name pfedme_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear --mu 15 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name pfedme_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear --mu 15 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# Ditto
python3 main.py --exp_name ditto_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name ditto_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# FedRep
python3 main.py --exp_name fedrep_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedrep_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear --nu 2 --mu 0.01 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name superfed-mm_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear --nu 2 --mu 0.01 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_mnist_noise_symmetric_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear --nu 2 --mu 0.01 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name superfed-lm_mnist_noise_symmetric_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear --nu 2 --mu 0.01 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &



## CIFAR10
# FedAvg
python3 main.py --exp_name fedavg_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedavg_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# FedProx
python3 main.py --exp_name fedprox_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedprox_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name lg-fedavg_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# FedPer
python3 main.py --exp_name fedper_cifar10_noise_symmetric_02 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedper_cifar10_noise_symmetric_06 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# APFL
python3 main.py --exp_name apfl_cifar10_noise_symmetric_02 --tb_port 4 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name apfl_cifar10_noise_symmetric_06 --tb_port 4 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# pFedMe
python3 main.py --exp_name pfedme_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm pfedme --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv --mu 15 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name pfedme_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm pfedme --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv --mu 15 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# Ditto
python3 main.py --exp_name ditto_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name ditto_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# FedRep
python3 main.py --exp_name fedrep_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name fedrep_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv --mu 0.01 --nu 2 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name superfed-mm_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar10_noise_symmetric_02 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv --mu 0.01 --nu 2 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &
python3 main.py --exp_name superfed-lm_cifar10_noise_symmetric_06 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv --mu 0.01 --nu 2 --L 0.2 \
--C 0.05 --K 100 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--eval_every 500 &