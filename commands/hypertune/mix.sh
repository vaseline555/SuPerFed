# SuPerFed-MM
python3 main.py --exp_name superfed-mm_mix00 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 1 --mu 0.01 --L 0 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_mix02 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 1 --mu 0.01 --L 0.2 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_mix04 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 1 --mu 0.01 --L 0.4 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_mix06 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 1 --mu 0.01 --L 0.6 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_mix08 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 1 --mu 0.01 --L 0.8 \
--split_type pathological --shard_size 250 \
--eval_every 100 &


# SuPerFed-LM
python3 main.py --exp_name superfed-lm_mix00 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0.01 --L 0.0 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_mix02 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0.01 --L 0.2 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_mix04 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0.01 --L 0.4 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_mix06 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0.01 --L 0.6 \
--split_type pathological --shard_size 250 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_mix08 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --R 300 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0.01 --L 0.8 \
--split_type pathological --shard_size 250 \
--eval_every 100
