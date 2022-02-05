### Set port number properly!!!

# P 0
python3 ../main.py P00 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.0 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15280 &

# P 0.1
python3 ../main.py P01 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.1 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15281 &

# P 0.2
python3 ../main.py P02 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.2 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15282 &

# P 0.3
python3 ../main.py P03 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.3 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15283 &

# P 0.4
python3 ../main.py P04 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15284 &

# P 0.5
python3 ../main.py P05 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.5 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15285 &

# P 0.6
python3 ../main.py P06 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.6 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15286 &

# P 0.7
python3 ../main.py P07 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.7 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15287 &

# P 0.8
python3 ../main.py P08 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.8 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15288 &

# P 0.9
python3 ../main.py P09 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.9 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 15289 &