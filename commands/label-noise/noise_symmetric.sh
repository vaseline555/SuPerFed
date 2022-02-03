# Label noise - Symmetric
## MNIST
python3 main.py --exp_name fedavg_mnist_iid_noise_sym_04 --tb_port 10555 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm fedavg --model_name ResNet18 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type iid \
--eval_every 100

python3 main.py --exp_name fedavg_mnist_iid_noise_sym_08 --tb_port 10556 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm fedavg --model_name ResNet18 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type iid \
--eval_every 100

python3 main.py --exp_name fedavg_mnist_patho_noise_sym_04 --tb_port 10557 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm fedavg --model_name ResNet18 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 100

python3 main.py --exp_name fedavg_mnist_patho_noise_sym_08 --tb_port 10558 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm fedavg --model_name ResNet18 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 100



## CIFAR10
python3 main.py --exp_name fedavg_cifar10_iid_noise_sym_04 --tb_port 14922 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type iid \
--eval_every 100

python3 main.py --exp_name fedavg_cifar10_iid_noise_sym_08 --tb_port 14923 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type iid \
--eval_every 100

python3 main.py --exp_name fedavg_cifar10_patho_noise_sym_04 --tb_port 7324 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 250 \
--eval_every 100

python3 main.py --exp_name fedavg_cifar10_patho_noise_sym_08 --tb_port 7325 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm fedavg --model_name MobileNetv2 \
--C 0.1 --K 100 --R 500 --E 5 --B 10 \
--split_type pathological --shard_size 250 \
--eval_every 100