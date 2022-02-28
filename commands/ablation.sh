# SuPerFed-MM
python3 main.py --exp_name superfed-mm_nu0mu0 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu0 --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu0 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu5mu0 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu10mu0 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu0mu01 --tb_port 4 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu01 --tb_port 5 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu01 --tb_port 6 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu5mu01 --tb_port 6 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu10mu01 --tb_port 6 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu0mu001 --tb_port 7 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu001 --tb_port 8 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu5mu001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu10mu001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu0mu0001 --tb_port 7 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu1mu0001 --tb_port 8 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu2mu0001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu5mu0001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-mm_nu10mu0001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0.001 \
--eval_every 100 &



# SuPerFed-LM
python3 main.py --exp_name superfed-lm_nu0mu0 --tb_port 1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu1mu0 --tb_port 2 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu2mu0 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu5mu0 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu10mu0 --tb_port 3 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu0mu01 --tb_port 4 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu1mu01 --tb_port 5 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu2mu01 --tb_port 6 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu5mu01 --tb_port 6 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu10mu01 --tb_port 6 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0.1 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu0mu001 --tb_port 7 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu1mu001 --tb_port 8 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu2mu001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu5mu001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu10mu001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0.01 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu0mu0001 --tb_port 7 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu1mu0001 --tb_port 8 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu2mu0001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu5mu0001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.001 \
--eval_every 100 &

python3 main.py --exp_name superfed-lm_nu10mu0001 --tb_port 9 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 20 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 10 --mu 0.001 \
--eval_every 100 &