### Set port number properly!!!

# mu 0 beta 0
python3 ../main.py MM_m0b0 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.0 --beta 0.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12594 &

# mu 0 beta 1
python3 ../main.py MM_m0b1 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.0 --beta 1.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12595 &

# mu 0 beta 2
python3 ../main.py MM_m0b2 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.0 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12596 &

# mu 0 beta 5
python3 ../main.py MM_m0b5 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0 --beta 5.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12597 &

# mu 0.01 beta 0
python3 ../main.py MM_m001b0 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 0.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12598 &

# mu 0.01 beta 1
python3 ../main.py MM_m001b1 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 1.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12599 &

# mu 0.01 beta 2
python3 ../main.py MM_m001b2 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12600 &

# mu 0.01 beta 5
python3 ../main.py MM_m001b5 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 5.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12601 &

# mu 0.1 beta 0
python3 ../main.py MM_m01b0 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.1 --beta 0.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12602 &

# mu 0.1 beta 1
python3 ../main.py MM_m01b1 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.1 --beta 1.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12603 &

# mu 0.1 beta 2
python3 ../main.py MM_m01b2 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.1 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12604 &

# mu 0.1 beta 5
python3 ../main.py MM_m01b5 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 0.1 --beta 5.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12605 &

# mu 1 beta 0 == FedProx
python3 ../main.py MM_m1b0 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 1.0 --beta 0.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12606 &

# mu 1 beta 1
python3 ../main.py MM_m1b1 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 1.0 --beta 1.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12607 &

# mu 1 beta 2
python3 ../main.py MM_m1b2 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 1.0 --beta 2.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12608 &

# mu 1 beta 5
python3 ../main.py MM_m1b5 \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 100 --E 10 --B 20 --L 0.4 --mu 1 --beta 5.0 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 12609 &