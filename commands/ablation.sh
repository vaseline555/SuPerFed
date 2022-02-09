# SuPerFed-MM
python3 main.py --exp_name superfed-mm_nu0 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0 --mu 0 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_nu01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0.1 --mu 0 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_nu1 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 1 --mu 0 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_nu05 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0.5 --mu 0 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_nu1 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 1 --mu 0 \
--eval_every 200 &

python3 main.py --exp_name superfed-mm_nu2 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 2 --mu 0 \
--eval_every 200 &


# SuPerFed-LM
python3 main.py --exp_name superfed-lm_0 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0 \
--eval_every 200

python3 main.py --exp_name superfed-lm_01 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 0.1\
--eval_every 200

python3 main.py --exp_name superfed-lm_1 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 1 \
--eval_every 200

python3 main.py --exp_name superfed-lm_2 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 2 \
--eval_every 200

python3 main.py --exp_name superfed-lm_5 --tb_port 1 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-lm --model_name TwoNN --fc_type LinesLinear \
--C 0.01 --K 500 --R 200 --E 5 --B 10 \
--split_type pathological --shard_size 60 --nu 5 \
--eval_every 200