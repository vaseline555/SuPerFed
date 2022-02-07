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

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_cifar100_diri_01 --tb_port 1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm lg-fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name lg-fedavg_cifar100_diri_05 --tb_port 2 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm lg-fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name lg-fedavg_cifar100_diri_1 --tb_port 3 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm lg-fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedPer
python3 main.py --exp_name fedper_cifar100_diri_01 --tb_port 2222 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedper --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name fedper_cifar100_diri_05 --tb_port 2223 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedper --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedper_cifar100_diri_1 --tb_port 2224 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedper --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# APFL
python3 main.py --exp_name apfl_cifar100_diri_01 --tb_port 7 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm apfl --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name apfl_cifar100_diri_05 --tb_port 8 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm apfl --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name apfl_cifar100_diri_1 --tb_port 9 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm apfl --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_cifar100_diri_01 --tb_port 1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm pfedme --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name pfedme_cifar100_diri_05 --tb_port 2 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm pfedme --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name pfedme_cifar100_diri_1 --tb_port 3 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm pfedme --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# Ditto
python3 main.py --exp_name ditto_cifar100_diri_01 --tb_port 1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm ditto --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name ditto_cifar100_diri_05 --tb_port 2 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm ditto --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name ditto_cifar100_diri_1 --tb_port 3 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm ditto --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedRep
python3 main.py --exp_name fedrep_cifar100_diri_01 --tb_port 1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedrep --model_name ResNet18 \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 1000
python3 main.py --exp_name fedrep_cifar100_diri_05 --tb_port 2 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedrep --model_name ResNet18 \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 1000
python3 main.py --exp_name fedrep_cifar100_diri_1 --tb_port 3 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedrep --model_name ResNet18 \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 1000

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

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_tin_diri_01 --tb_port 2  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm lg-fedavg --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name lg-fedavg_tin_diri_05 --tb_port 3 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm lg-fedavg --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name lg-fedavg_tin_diri_1 --tb_port 4 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm lg-fedavg --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedPer
python3 main.py --exp_name fedper_tin_diri_01 --tb_port 1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedper --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name fedper_tin_diri_05 --tb_port 2 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name fedper_tin_diri_1 --tb_port 3 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedper --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# Ditto
python3 main.py --exp_name ditto_tin_diri_01 --tb_port 2 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm ditto --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 200
python3 main.py --exp_name ditto_tin_diri_05 --tb_port 3 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm ditto --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 200
python3 main.py --exp_name ditto_tin_diri_1 --tb_port 4 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm ditto --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 200

# FedRep
python3 main.py --exp_name fedrep_tin_diri_01 --tb_port 1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedrep --model_name MobileNetv2 \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 1000
python3 main.py --exp_name fedrep_tin_diri_05 --tb_port 2 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedrep --model_name MobileNetv2 \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 1000
python3 main.py --exp_name fedrep_tin_diri_1 --tb_port 3 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedrep --model_name MobileNetv2 \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 1000

# APFL
python3 main.py --exp_name apfl_tin_diri_01 --tb_port 1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm apfl --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 1000
python3 main.py --exp_name apfl_tin_diri_05 --tb_port 2 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm apfl --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 1000
python3 main.py --exp_name apfl_tin_diri_1 --tb_port 3 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm apfl --model_name MobileNetv2 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.01 --K 500 --R 1000 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 1000

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