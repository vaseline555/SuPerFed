## SVHN
# FedAvg
python3 main.py --exp_name fedavg_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name VGG9 \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# FedProx
python3 main.py --exp_name fedprox_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name VGG9 \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# Ditto
python3 main.py --exp_name ditto_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name VGG9 --fc_type LinesLinear --conv_type LinesConv \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# APFL
python3 main.py --exp_name apfl_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name VGG9 --fc_type LinesLinear --conv_type LinesConv \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# pFedMe
python3 main.py --exp_name pfedme_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm pfedme --model_name VGG9 --fc_type LinesLinear --conv_type LinesConv --mu 15 \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name VGG9 --fc_type LinesLinear --conv_type LinesConv --mu 0.01 --nu 2 --L 0.2 \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_svhn_holdout --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name VGG9 --fc_type LinesLinear --conv_type LinesConv --mu 0.01 --nu 2 --L 0.2 \
--C 0.012 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 10.0 \
--evaluate_on_holdout_clients --eval_every 500 &



## Caltech101
# FedAvg
python3 main.py --exp_name fedavg_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name ResNet18 \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# FedProx
python3 main.py --exp_name fedprox_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name ResNet18 \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# Ditto
python3 main.py --exp_name ditto_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# APFL
python3 main.py --exp_name apfl_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# pFedMe
python3 main.py --exp_name pfedme_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN --mu 15 \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN --mu 0.01 --nu 2 --L 0.2 \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_caltech_holdout --tb_port 3 \
--dataset MNIST --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN --mu 0.01 --nu 2 --L 0.2 \
--C 0.06 --K 500 --R 500 --E 10 --B 20 \
--split_type dirichlet --alpha 1.0 \
--evaluate_on_holdout_clients --eval_every 500 &