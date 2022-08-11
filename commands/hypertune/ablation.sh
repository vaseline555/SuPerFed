# SuPerFed-MM
python3 main.py --exp_name superfed-mm_nu0mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu1mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu2mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu5mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0 \
--eval_every 20

python3 main.py --exp_name superfed-mm_nu0mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 1 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu1mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 1 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu2mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 1 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu5mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 1 \
--eval_every 20

python3 main.py --exp_name superfed-mm_nu0mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.1 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu1mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.1 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu2mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.1 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu5mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.1 \
--eval_every 20

python3 main.py --exp_name superfed-mm_nu0mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.01 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu1mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.01 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu2mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.01 \
--eval_every 20 &

python3 main.py --exp_name superfed-mm_nu5mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-mm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.01 \
--eval_every 20



# SuPerFed-LM
python3 main.py --exp_name superfed-lm_nu0mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu1mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu2mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu5mu0 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0 \
--eval_every 20

python3 main.py --exp_name superfed-lm_nu0mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 1 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu1mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 1 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu2mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 1 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu5mu1 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 1 \
--eval_every 20

python3 main.py --exp_name superfed-lm_nu0mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.1 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu1mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.1 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu2mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.1 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu5mu01 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.1 \
--eval_every 20

python3 main.py --exp_name superfed-lm_nu0mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 0 --mu 0.01 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu1mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.1 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 1 --mu 0.01 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu2mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 2 --mu 0.01 \
--eval_every 20 &

python3 main.py --exp_name superfed-lm_nu5mu001 \
--dataset CIFAR10 --is_small --in_channels 3 --num_classes 10 \
--algorithm superfed-lm --model_name TwoCNN --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.05 --K 100 --R 100 --E 10 --B 10 --lr 0.01 --L 0.0 \
--split_type pathological --shard_size 250 --nu 5 --mu 0.01 \
--eval_every 20 
