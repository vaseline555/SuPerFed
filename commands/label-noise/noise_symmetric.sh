# Label noise - Symmetric
## MNIST
python3 main.py --exp_name _mnist_patho_noise_sym_02 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

python3 main.py --exp_name _mnist_patho_noise_sym_06 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedavg_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# FedProx
python3 main.py --exp_name fedprox_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedprox_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedprox --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name lg-fedavg_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm lg-fedavg --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# FedPer
python3 main.py --exp_name fedper_mnist_patho_noise_sym_02 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedper_mnist_patho_noise_sym_06 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedper --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# APFL
python3 main.py --exp_name apfl_mnist_patho_noise_sym_02 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm apfl --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name apfl_mnist_patho_noise_sym_06 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm apfl  --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 --mu 15 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name pfedme_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm pfedme --model_name TwoNN --mu 15 --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# Ditto
python3 main.py --exp_name ditto_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name ditto_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# FedRep
python3 main.py --exp_name fedrep_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedrep_mnist_patho_noise_sym_06 --tb_port 2  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedrep --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-mm_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_mnist_patho_noise_sym_02 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-lm_mnist_patho_noise_sym_06 --tb_port 2 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200



## CIFAR10
python3 main.py --exp_name _cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm  --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200

python3 main.py --exp_name _cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm  --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedAvg
python3 main.py --exp_name _cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name fedavg_cifar10_patho_noise_pair_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedProx
python3 main.py --exp_name fedprox_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name fedprox_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedprox --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_cifar10_patho_noise_pair_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name lg-fedavg_cifar10_patho_noise_pair_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm lg-fedavg --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedPer
python3 main.py --exp_name fedper_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name fedper_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedper --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# APFL
python3 main.py --exp_name apfl_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name apfl_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm pfedme --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 --mu 15 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name pfedme_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm pfedme --model_name TwoCNN --mu 15 --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# Ditto
python3 main.py --exp_name ditto_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv  \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name ditto_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv  \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# FedRep
python3 main.py --exp_name fedrep_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name fedrep_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm fedrep --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-mm  --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name superfed-mm_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar10_patho_noise_sym_02 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.2 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5  \
--eval_every 200
python3 main.py --exp_name superfed-lm_cifar10_patho_noise_sym_06 --tb_port 28247 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.6 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200