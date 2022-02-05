# Dirichlet distribution based non-IID setting
## FEMNIST
python3 main.py --exp_name _femnist --tb_port  \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm  --model_name ResNet18 \
--C 0.01 --R 200 --E 5 --B 10 \
--split_type realistic \
--eval_every 200



## Shakespeare
python3 main.py --exp_name _shakespeare --tb_port \
--dataset Shakespeare --num_classes 100 \
--algorithm  --model_name NextCharLM \
--C 0.015 --R 200 --E 5 --B 10 --mu 0.01 \
--split_type realistic \
--eval_every 200
