# SuPerFed-MM
python3 main.py --exp_name superfed-lm_L0_mu0_nu_10 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.0 \
--split_type pathological --shard_size 60 --mu 0 --nu 10 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L0_mu0_nu10 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.0 \
--split_type pathological --shard_size 60 --mu 0 --nu 10 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L2 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.2 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L3 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.3 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L4 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.4 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L5 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.5 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L6 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.6 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L7 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.7 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_L8 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 --L 0.8 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200 &




# SuPerFed-LM
python3 main.py --exp_name superfed-lm_0 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0 \
--eval_every 200

