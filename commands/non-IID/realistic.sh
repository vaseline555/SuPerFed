# Dirichlet distribution based non-IID setting
###########
# FEMNIST #
###########
# FedAvg
python3 main.py --exp_name fedavg_femnist --tb_port 11 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedavg --model_name VGG9 \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# FedProx
python3 main.py --exp_name fedprox_femnist --tb_port 12 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedprox --model_name VGG9 --mu 0.01 \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# SCAFFOLD
python3 main.py --exp_name scaffold_femnist --tb_port 2222 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm scaffold --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_femnist --tb_port 13 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm lg-fedavg --model_name VGG9 \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# FedPer
python3 main.py --exp_name fedper_femnist --tb_port 14 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedper --model_name VGG9 \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# APFL
python3 main.py --exp_name apfl_femnist --tb_port 2222 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm apfl --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# pFedMe
python3 main.py --exp_name pfedme_femnist --tb_port 2222 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm pfedme --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --mu 10 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_femnist --tb_port 2222 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm ditto --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --tau 10 --mu 1 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# FedRep
python3 main.py --exp_name fedrep_femnist --tb_port 2222 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedrep --model_name VGG9 \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --tau 10 --n_jobs 10 \
--split_type realistic \
--eval_every 300 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_femnist --tb_port 6666 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm superfed-mm --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --nu 0.5 --mu 0.01 --L 0.4 \
--split_type realistic --n_jobs 10 \
--eval_every 100

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_femnist --tb_port 6667 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm superfed-lm --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.014 --R 300 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0 --L 0.4 \
--split_type realistic --n_jobs 10 \
--eval_every 100


###############
# Shakespeare #
###############
# FedAvg
python3 main.py --exp_name fedavg_shakespeare --tb_port 6000 \
--dataset Shakespeare \
--algorithm fedavg --model_name NextCharLM \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_shakespeare --tb_port 6001 \
--dataset Shakespeare \
--algorithm fedprox --model_name NextCharLM \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --mu 0.01 --n_jobs 10 \
--split_type realistic --n_jobs 10 \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_shakespeare --tb_port 2222 \
--dataset Shakespeare \
--algorithm scaffold --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_shakespeare --tb_port 6002 \
--dataset Shakespeare \
--algorithm lg-fedavg --model_name NextCharLM \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_shakespeare --tb_port 6003 \
--dataset Shakespeare \
--algorithm fedper --model_name NextCharLM \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# APFL
python3 main.py --exp_name apfl_shakespeare --tb_port 6004 \
--dataset Shakespeare \
--algorithm apfl --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_shakespeare --tb_port 6005 \
--dataset Shakespeare \
--algorithm pfedme --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --mu 1 --n_jobs 10 \
--split_type realistic \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_shakespeare --tb_port 6006 \
--dataset Shakespeare \
--algorithm fedrep --model_name NextCharLM \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 --tau 1 \
--split_type realistic \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_shakespeare --tb_port 6007 \
--dataset Shakespeare \
--algorithm ditto --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --n_jobs 10 --tau 1 --mu 1 \
--split_type realistic \
--eval_every 100 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_shakespeare --tb_port 6747 \
--dataset Shakespeare \
--algorithm superfed-mm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding  \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --nu 5 --mu 0 --n_jobs 10 --L 0.4 \
--split_type realistic \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_shakespeare --tb_port 6748 \
--dataset Shakespeare \
--algorithm superfed-lm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.016 --R 200 --E 1 --B 50 --lr 0.8 --nu 2 --mu 0 --n_jobs 10 --L 0.4 \
--split_type realistic \
--eval_every 50 &