# Dirichlet distribution based non-IID setting
############
# CIFAR100 #
############
# FedAvg
python3 main.py --exp_name fedavg_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedavg_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedavg_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedprox --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 --mu 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedprox_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedprox --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 --mu 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedprox_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedprox --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 --mu 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm scaffold --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 390 --E 5 --B 20 --lr 0.01 --lr_decay 0.9 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name scaffold_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm scaffold --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 390 --E 5 --B 20 --lr 0.01 --lr_decay 0.9 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name scaffold_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm scaffold --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 390 --E 5 --B 20 --lr 0.01 --lr_decay 0.9 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm lg-fedavg --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm lg-fedavg --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm lg-fedavg --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedper --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedper_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedper --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedper_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedper --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# APFL
python3 main.py --exp_name apfl_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm apfl --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name apfl_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm apfl --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name apfl_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm apfl --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm pfedme --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --mu 15 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name pfedme_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm pfedme --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --mu 15 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name pfedme_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm pfedme --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --mu 15 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm ditto --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --tau 5 --mu 1 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name ditto_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm ditto --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --tau 5 --mu 1 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name ditto_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm ditto --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --tau 5 --mu 1 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedrep --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --tau 5 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedrep_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedrep --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --tau 5 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name fedrep_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedrep --model_name ResNet9 \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --tau 5 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-mm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.4 --nu 1 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 1 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-mm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.4 --nu 1 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 10 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-mm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.4 --nu 1 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.6 --nu 2 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_cifar100_diri_10 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.6 --nu 2 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 10.0 --n_jobs 20 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_cifar100_diri_100 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.6 --nu 2 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 100.0 --n_jobs 20 \
--eval_every 50 &



################
# TinyImageNet #
################
# FedAvg
python3 main.py --exp_name fedavg_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedavg --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name fedavg_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedavg --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name fedavg_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedavg --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedprox --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --mu 0.01 --n_jobs 20 --lr 0.01 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name fedprox_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedprox --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --mu 0.01 --n_jobs 20 --lr 0.01 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name fedprox_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedprox --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --mu 0.01 --n_jobs 20 --lr 0.01 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm scaffold --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name scaffold_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm scaffold --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name scaffold_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm scaffold --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm lg-fedavg --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm lg-fedavg --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name lg-fedavg_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm lg-fedavg --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedper --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name fedper_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedper --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name fedper_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedper --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# APFL
python3 main.py --exp_name apfl_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm apfl --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name apfl_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm apfl --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name apfl_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm apfl --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_tin_diri_1  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm pfedme --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --mu 15 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name pfedme_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm pfedme --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --mu 15 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name pfedme_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm pfedme --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --mu 15 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm ditto --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --tau 5 --mu 1 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name ditto_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm ditto --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --tau 5 --mu 1 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name ditto_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm ditto --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --n_jobs 20 --tau 5 --mu 1 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedrep --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --tau 5 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name fedrep_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedrep --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --tau 5 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name fedrep_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm fedrep --model_name MobileNet \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --tau 5 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-mm --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --L 0.35 --nu 1 --mu 0.01 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-mm --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --L 0.45 --nu 1 --mu 0.01 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name superfed-mm_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-mm --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --L 0.55 --nu 1 --mu 0.01 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_tin_diri_1 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-lm --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --L 0.45 --nu 2 --mu 0.01 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 1.0 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_tin_diri_10 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-lm --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --L 0.5 --nu 2 --mu 0.01 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 10.0 \
--eval_every 50 &
python3 main.py --exp_name superfed-lm_tin_diri_100 \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm superfed-lm --model_name MobileNet --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.025 --K 200 --R 500 --E 5 --B 20 --L 0.55 --nu 2 --mu 0.01 --n_jobs 20 --lr 0.02 \
--split_type dirichlet --alpha 100.0 \
--eval_every 50 &
