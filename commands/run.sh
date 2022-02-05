### Set port number properly!!!

# Run SuPerFed_MM with personalization - CIFAR10
python3 ../main.py SuPerFed_MM \
--global_seed 5959 --device cuda --dataset CIFAR10 --iid 0 --num_shards 200 \
--C 0.1 --K 100 --R 200 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 2 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 10 --tb_port 18365 &

# Run FedProx with personalization - CIFAR100
python3 ../main.py SuPerFed_MM \
--global_seed 5959 --device cuda --dataset CIFAR100 --iid 0 --num_shards 1000 \
--C 0.1 --K 200 --R 300 --E 10 --B 20 --L 0.5 --mu 0.01 --beta 2 \
--init_seed 595959 525252 \
--model_name CIFARConvNet --in_channels 3 --num_classes 100 --tb_port 7988 &

# Run FedProx with personalization - EMNIST
python3 ../main.py SuPerFed_MM \
--global_seed 5959 --device cuda --dataset EMNIST --iid 0 \
--C 0.01 --K 1543 --R 100 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 2 \
--init_seed 595959 525252 \
--model_name MNISTConvNet --in_channels 1 --num_classes 62 --tb_port 18367 &

# Run FedProx with personalization - TinyImageNet
python3 ../main.py SuPerFed_MM \
--global_seed 5959 --device cuda --dataset TinyImageNet --iid 0 \
--C 0.03 --K 388 --R 500 --E 10 --B 20 --L 0.4 --mu 0.01 --beta 2 \
--init_seed 595959 525252 \
--model_name TINConvNet --in_channels 3 --num_classes 200 --tb_port 18368 &