# Dirichlet distribution based non-IID setting
## CIFAR100
python3 main.py --exp_name fedavg_cifar100_diri_01 --tb_port  \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm  --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 100

python3 main.py --exp_name fedavg_cifar100_diri_05 --tb_port  \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm  --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 100

python3 main.py --exp_name fedavg_cifar100_diri_1 --tb_port  \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm fedavg --model_name ResNet18 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 100



## TinyImageNet
python3 main.py --exp_name fedavg_tin_diri_01 --tb_port  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.1 \
--eval_every 100

python3 main.py --exp_name fedavg_tin_diri_05 --tb_port  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 0.5 \
--eval_every 100

python3 main.py --exp_name fedavg_tin_diri_1 --tb_port  \
--dataset TinyImageNet --in_channels 3 --num_classes 200 \
--algorithm  --model_name MobileNetv2 \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type dirichlet --alpha 1.0 \
--eval_every 100