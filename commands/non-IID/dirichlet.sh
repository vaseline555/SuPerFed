# Dirichlet distribution based non-IID setting
############
# CIFAR100 #
############
python3 main.py --exp_name _cifar100_diri_01 --tb_port  \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm  --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200

python3 main.py --exp_name _cifar100_diri_05 --tb_port  \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm  --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

python3 main.py --exp_name _cifar100_diri_1 --tb_port  \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm  --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_cifar100_diri_01 --tb_port 7570 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name fedavg_cifar100_diri_05 --tb_port 7571 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedavg_cifar100_diri_1 --tb_port 7572 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedProx
python3 main.py --exp_name fedprox_cifar100_diri_01 --tb_port 23453 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedprox --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name fedprox_cifar100_diri_05 --tb_port 23454 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedprox --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedprox_cifar100_diri_1 --tb_port 23455 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedprox --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar100_diri_01 --tb_port 20490 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-mm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name superfed-mm_cifar100_diri_05 --tb_port 20491 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-mm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-mm_cifar100_diri_1 --tb_port 20492 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-mm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar100_diri_01 --tb_port 14225 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name superfed-lm_cifar100_diri_05 --tb_port 14226 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-lm_cifar100_diri_1 --tb_port 14227 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200



################
# TinyImageNet #
################
python3 main.py --exp_name _tin_diri_01 --tb_port  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200

python3 main.py --exp_name _tin_diri_05 --tb_port  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200

python3 main.py --exp_name _tin_diri_1 --tb_port  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_tin_diri_01 --tb_port 7573 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name fedavg_tin_diri_05 --tb_port 7574 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedavg_tin_diri_1 --tb_port 7575 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedProx
python3 main.py --exp_name fedprox_tin_diri_01 --tb_port 23456 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedprox --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name fedprox_tin_diri_05 --tb_port 23457 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedprox --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedprox_tin_diri_1 --tb_port 23458 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedprox --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_tin_diri_01 --tb_port 20493 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-mm --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name superfed-mm_tin_diri_05 --tb_port 20494 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-mm --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-mm_tin_diri_1 --tb_port 20495 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-mm --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_tin_diri_01 --tb_port 20499 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-lm --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name superfed-lm_tin_diri_05 --tb_port 20500 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-lm --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name superfed-lm_tin_diri_1 --tb_port 20501 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-lm --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200