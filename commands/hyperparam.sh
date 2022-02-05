### Set port number properly!!!

# E 1 B 4
python3 ../main.py MM_E1B4 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 1 --B 4 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4279 &

# E 5 B 4
python3 ../main.py MM_E5B4 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 5 --B 4 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4280 &

# E 10 B 4
python3 ../main.py MM_E10B4 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 4 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4281 &

# E 20 B 4
python3 ../main.py MM_E20B4 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 20 --B 4 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4282 &

# E 1 B 10
python3 ../main.py MM_E1B10 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 1 --B 10 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4283 &

# E 5 B 10
python3 ../main.py MM_E5B10 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 5 --B 10 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4284 &

# E 10 B 10
python3 ../main.py MM_E10B10 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 10 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4285 &

# E 20 B 10
python3 ../main.py MM_E20B10 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 20 --B 10 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4286 &

# E 1 B 20
python3 ../main.py MM_E1B20 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 1 --B 20 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4287 &

# E 5 B 20
python3 ../main.py MM_E5B20 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 5 --B 20 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4288 &

# E 10 B 20
python3 ../main.py MM_E10B20 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4289 &

# E 20 B 20
python3 ../main.py MM_E20B20 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 20 --B 20 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4290 &

# E 1 B 40
python3 ../main.py MM_E1B40 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 1 --B 40 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4291 &

# E 5 B 40
python3 ../main.py MM_E5B40 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 5 --B 40 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4292 &

# E 10 B 40
python3 ../main.py MM_E10B40 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 40 --L 0.4 --mu 0.01 --beta 22.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4293 &

# E 20 B 40
python3 ../main.py MM_E20B40 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 20 --B 40 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 4294 &