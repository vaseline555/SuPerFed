# Label noise - Pair
## MNIST
python3 main.py --exp_name _mnist_diri_noise_pair_01 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

python3 main.py --exp_name _mnist_patho_noise_pair_04 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedavg_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedProx
python3 main.py --exp_name fedprox_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedprox_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name lg-fedavg_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedPer
python3 main.py --exp_name fedper_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedper_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# APFL
python3 main.py --exp_name apfl_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name apfl_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name pfedme_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# Ditto
python3 main.py --exp_name ditto_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear  \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name ditto_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear  \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedRep
python3 main.py --exp_name fedrep_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedrep_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-mm_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_mnist_diri_noise_pair_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-lm_mnist_patho_noise_pair_04 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200



## CIFAR10
python3 main.py --exp_name _cifar10_patho_noise_sym_01 --tb_port \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm  --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

python3 main.py --exp_name _cifar10_patho_noise_sym_04 --tb_port \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm  --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name _cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedProx
python3 main.py --exp_name fedprox_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedprox_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# LG-FedAvg
python3 main.py --exp_name lg-fed_avg_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm lg-fed_avg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name lg-fed_avg_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm lg-fed_avg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedPer
python3 main.py --exp_name fedper_cifar10_patho_noise_pair_01 --tb_port \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedper_cifar10_patho_noise_pair_04 --tb_port \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# APFL
python3 main.py --exp_name apfl_cifar10_patho_noise_pair_01 --tb_port \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name apfl_cifar10_patho_noise_pair_04 --tb_port \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm pfedme --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name pfedme_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm pfedme --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# Ditto
python3 main.py --exp_name ditto_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name ditto_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedRep
python3 main.py --exp_name fedrep_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedrep_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm superfed-mm --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-mm_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm superfed-mm --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar10_patho_noise_pair_01 --tb_port 11 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.1 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-lm_cifar10_patho_noise_pair_04 --tb_port 22 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type pair --noise_rate 0.4 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200