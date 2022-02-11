# SuPerFed-MM
python3 main.py --exp_name superfed-mm_nu0mu0 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 0 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu0 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 1 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu0 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 2 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu0mu01 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu01 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu01 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu0mu001 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu001 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu001 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.2 \
--split_type pathological --shard_size 250--nu 2 --mu 0.01 \
--eval_every 100 &


# SuPerFed-LM
python3 main.py --exp_name superfed-lm_0 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 100 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0 \
--eval_every 200

python3 main.py --exp_name superfed-lm_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 100 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0.1\
--eval_every 200

python3 main.py --exp_name superfed-lm_1 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 100 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 1 \
--eval_every 200

python3 main.py --exp_name superfed-lm_2 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 100 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200

python3 main.py --exp_name superfed-lm_5 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.05 --K 100 --R 100--E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 5 \
--eval_every 200