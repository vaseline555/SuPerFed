# Dirichlet distribution based non-IID setting
## FEMNIST
python3 main.py --exp_name fedavg_femnist --tb_port 20944 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedavg --model_name ResNet18 \
--C 0.002 --R 500 --E 5 --B 10 \
--split_type realistic \
--eval_every 100



## Shakespeare
python3 main.py --exp_name test --tb_port 20346 \
--dataset Shakespeare --num_classes 100 \
--algorithm fedavg --model_name NextCharLM \
--C 0.015 --R 500 --E 5 --B 10 --mu 0.01 \
--split_type realistic \
--eval_every 100

python3 main.py --exp_name apfl_shakespeare --tb_port 20346 \
--dataset Shakespeare --num_classes 100 \
--algorithm apfl --model_name NextCharLM \
--C 0.05 --R 50 --E 5 --B 10 --mu 0.01 \
--split_type realistic \
--fc_type LinesLinear --embedding_type LinesEmbedding --lstm_type StandardLSTM \
--eval_every 50