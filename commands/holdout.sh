## KMNIST
python3 main.py --exp_name _kmnist_holdout --tb_port \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm --model_name TwoNN \
--C 0.006 --K 1000 --R 50 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 50

# FedAvg
python3 main.py --exp_name fedavg_kmnist_holdout --tb_port 1 \
--dataset KMNIST --in_channels 1 --num_classes 10 \
--algorithm fedavg --model_name TwoNN \
--C 0.006 --K 1000 --R 50 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 50



## SVHN
python3 main.py --exp_name _svhn_holdout --tb_port \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm --model_name TwoCNN \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100

# FedAvg
python3 main.py --exp_name fedavg_svhn_holdout --tb_port 2 \
--dataset SVHN --is_small --in_channels 3 --num_classes 10 \
--algorithm fedavg --model_name TwoCNN \
--C 0.006 --K 1000 --R 100 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--evaluate_on_holdout_clients --eval_every 100



## Places365
python3 main.py --exp_name _places365_holdout --tb_port \
--dataset Places365 --in_channels 3 --num_classes 365 \
--algorithm --model_name ResNet18 \
--C 0.006 --K 1000 --R 50 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--evaluate_on_holdout_clients --eval_every 50