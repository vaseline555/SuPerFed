# Dirichlet distribution based non-IID setting
###########
# FEMNIST #
###########
# FedAvg
python3 main.py --exp_name fedavg_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedavg --model_name VGG9 \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedprox --model_name VGG9 --mu 0.01 \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --mu 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm scaffold --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm lg-fedavg --model_name VGG9 \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedper --model_name VGG9 \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50

# APFL
python3 main.py --exp_name apfl_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm apfl --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm pfedme --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --n_jobs 10 --mu 5 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50

# Ditto
python3 main.py --exp_name ditto_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm ditto --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --tau 5 --mu 1 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm fedrep --model_name VGG9 \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --tau 5 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm superfed-mm --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --nu 2 --mu 0.01 --L 0.25 --lr_decay 0.995 \
--split_type realistic --n_jobs 10 \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_femnist \
--dataset FEMNIST --is_small --in_channels 1 --num_classes 62 \
--algorithm superfed-lm --model_name VGG9 --fc_type LinesLinear --bn_type LinesBN --conv_type LinesConv \
--C 0.007 --R 500 --E 10 --B 10 --lr 0.01 --nu 3 --mu 0.01 --L 0.45 --lr_decay 0.995 \
--split_type realistic --n_jobs 10 \
--eval_every 50 & 



###############
# Shakespeare #
###############
# FedAvg
python3 main.py --exp_name fedavg_shakespeare \
--dataset Shakespeare \
--algorithm fedavg --model_name NextCharLM \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# FedProx
python3 main.py --exp_name fedprox_shakespeare \
--dataset Shakespeare \
--algorithm fedprox --model_name NextCharLM \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --mu 0.01 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic --n_jobs 10 \
--eval_every 50 &

# SCAFFOLD
python3 main.py --exp_name scaffold_shakespeare \
--dataset Shakespeare \
--algorithm scaffold --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# LG-FedAvg
python3 main.py --exp_name lg-fedavg_shakespeare \
--dataset Shakespeare \
--algorithm lg-fedavg --model_name NextCharLM \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# FedPer
python3 main.py --exp_name fedper_shakespeare \
--dataset Shakespeare \
--algorithm fedper --model_name NextCharLM \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# APFL
python3 main.py --exp_name apfl_shakespeare \
--dataset Shakespeare \
--algorithm apfl --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# pFedMe
python3 main.py --exp_name pfedme_shakespeare \
--dataset Shakespeare \
--algorithm pfedme --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --mu 0.1 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# FedRep
python3 main.py --exp_name fedrep_shakespeare \
--dataset Shakespeare \
--algorithm fedrep --model_name NextCharLM \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --tau 5 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# Ditto
python3 main.py --exp_name ditto_shakespeare \
--dataset Shakespeare \
--algorithm ditto --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --tau 5 --mu 1 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# SuPerFed-MM
python3 main.py --exp_name superfed-mm_shakespeare \
--dataset Shakespeare \
--algorithm superfed-mm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --nu 1.5 --mu 0.01 --L 0.45 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &

# SuPerFed-LM
python3 main.py --exp_name superfed-lm_shakespeare \
--dataset Shakespeare \
--algorithm superfed-lm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --nu 2 --mu 0.01 --L 0.35 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50 &
