# Label noise - Symmetric
## MNIST
python3 main.py --exp_name _mnist_iid_noise_pair_04 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type iid \
--eval_every 200

python3 main.py --exp_name _mnist_iid_noise_pair_08 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type iid \
--eval_every 200

python3 main.py --exp_name _mnist_patho_noise_pair_04 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 200

python3 main.py --exp_name _mnist_patho_noise_pair_08 --tb_port  \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm  --model_name TwoNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 300 \
--eval_every 200



## CIFAR10
python3 main.py --exp_name _cifar10_iid_noise_pair_04 --tb_port  \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm  --model_name TwoCNN \
--C 0.05 --K 100 --R 200 --E 5 --B 10 \
--split_type iid \
--eval_every 200

python3 main.py --exp_name _cifar10_iid_noise_pair_08 --tb_port  \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm  --model_name TwoCNN \
--C 0.1 --K 100 --R 200 --E 5 --B 10 \
--split_type iid \
--eval_every 200

python3 main.py --exp_name _cifar10_patho_noise_pair_04 --tb_port 28246 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.4 \
--algorithm  --model_name TwoCNN \
--C 0.1 --K 100 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 250 \
--eval_every 200

python3 main.py --exp_name _cifar10_patho_noise_pair_08 --tb_port 28247 \
--dataset  --is_small --in_channels 3 --num_classes 10 \
--label_noise --noise_type symmetric --noise_rate 0.8 \
--algorithm fedavg --model_name TwoCNN \
--C 0.1 --K 100 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 250 \
--eval_every 200