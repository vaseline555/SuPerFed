## KMNIST
python3 main.py --exp_name _kmnist_holdout --tb_port \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm --model_name TwoNN \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100

# FedAvg
python3 main.py --exp_name fedavg_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100

# FedProx
python3 main.py --exp_name fedprox_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm fedprox --model_name TwoNN \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100

# Ditto
python3 main.py --exp_name ditto_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm ditto --model_name TwoNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100

# APFL
python3 main.py --exp_name apfl_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm apfl --model_name TwoNN  --fc_type LinesLinear \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100 

# pFedMe
python3 main.py --exp_name pfedme_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm pfedme --model_name TwoNN --fc_type LinesLinear \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 --mu 15 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 --mu 0 --nu 0.5 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100



## SVHN
python3 main.py --exp_name _svhn_holdout --tb_port \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm --model_name TwoCNN \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200

# FedProx
python3 main.py --exp_name fedprox_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm fedprox --model_name TwoCNN \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200

# Ditto
python3 main.py --exp_name ditto_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm ditto --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200

# APFL
python3 main.py --exp_name apfl_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm apfl --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm pfedme --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 --mu 15 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --conv_type LinesConv \
--C 0.006 --K 1000 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 200




## Caltech101
python3 main.py --exp_name _caltech_holdout --tb_port \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm --model_name ResNet18 \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200

# FedAvg
python3 main.py --exp_name fedavg_caltech_holdout --tb_port 3 \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm fedavg --model_name ResNet18 \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200

# FedProx
python3 main.py --exp_name fedprox_caltech_holdout --tb_port 3 \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm fedprox --model_name ResNet18 \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200

# Ditto
python3 main.py --exp_name ditto_caltech_holdout --tb_port 3 \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm ditto --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200

# APFL
python3 main.py --exp_name apfl_caltech_holdout --tb_port 3 \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm apfl --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200

# pFedMe
python3 main.py --exp_name pfedme_caltech_holdout --tb_port 3 \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm pfedme --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_caltech_holdout --tb_port 3 \
--dataset Caltech101 --in_channels 1 --num_classes 101 \
--algorithm superfed-mm --model_name ResNet18 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.006 --K 100 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 200