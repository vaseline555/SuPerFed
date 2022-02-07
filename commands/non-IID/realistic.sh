# Dirichlet distribution based non-IID setting
###########
# FEMNIST #
###########
python3 main.py --exp_name _femnist --tb_port  \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm  --model_name VGG9 \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001 \
--split_type realistic \
--eval_every 1000

# FedAvg
python3 main.py --exp_name fedavg_femnist --tb_port 11 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedavg --model_name VGG9 \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001  \
--split_type realistic \
--eval_every 1000

# FedProx
python3 main.py --exp_name fedprox_femnist --tb_port 12 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedprox --model_name VGG9 \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001  \
--split_type realistic \
--eval_every 1000

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_femnist --tb_port 13 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm lg-fedavg --model_name VGG9 \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001  \
--split_type realistic \
--eval_every 1000

# FedPer
python3 main.py --exp_name fedper_femnist --tb_port 14 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedper --model_name VGG9 \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001  \
--split_type realistic \
--eval_every 1000

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_femnist --tb_port 6666 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm superfed-mm --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001  \
--split_type realistic \
--eval_every 1000

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_femnist --tb_port 6667 \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm superfed-lm --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 1000 --E 5 --B 10 --lr 0.001  \
--split_type realistic \
--eval_every 1000


###############
# Shakespeare #
##############
python3 main.py --exp_name _shakespeare --tb_port \
--dataset Shakespeare \
--algorithm  --model_name NextCharLM \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50

# FedAvg
python3 main.py --exp_name fedavg_shakespeare --tb_port 6000 \
--dataset Shakespeare \
--algorithm fedavg --model_name NextCharLM \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# FedProx
python3 main.py --exp_name fedprox_shakespeare --tb_port 6001 \
--dataset Shakespeare \
--algorithm fedprox --model_name NextCharLM \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_shakespeare --tb_port 6002 \
--dataset Shakespeare \
--algorithm lg-fedavg --model_name NextCharLM \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0  \
--split_type realistic \
--eval_every 50 --n_jobs 5

# FedPer
python3 main.py --exp_name fedper_shakespeare --tb_port 6003 \
--dataset Shakespeare \
--algorithm fedper --model_name NextCharLM \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# APFL
python3 main.py --exp_name apfl_shakespeare --tb_port 6004 \
--dataset Shakespeare \
--algorithm apfl --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# pFedMe
python3 main.py --exp_name pfedme_shakespeare --tb_port 6005 \
--dataset Shakespeare \
--algorithm pfedme --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# FedRep
python3 main.py --exp_name fedrep_shakespeare --tb_port 6006 \
--dataset Shakespeare \
--algorithm fedrep --model_name NextCharLM \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# Ditto
python3 main.py --exp_name ditto_shakespeare --tb_port 6007 \
--dataset Shakespeare \
--algorithm ditto --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_shakespeare --tb_port 6747 \
--dataset Shakespeare --num_classes 100 \
--algorithm superfed-mm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_shakespeare --tb_port 6748 \
--dataset Shakespeare --num_classes 100 \
--algorithm superfed-lm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 50 --E 1 --B 500 --mu 0.01 --lr 1.0 \
--split_type realistic \
--eval_every 50 --n_jobs 5