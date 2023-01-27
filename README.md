# (SuPerFed) Connecting Low-Loss Subspace for Personalized Federated Learning [![Generic badge](https://img.shields.io/badge/code-official-green.svg)](https://shields.io/)
![SuPerFed_Overview](https://github.com/vaseline555/SuPerFed/blob/main/assets/SuPerFed_Overview.jpg)
This repository is an official implementation of the SIGKDD 2022 paper `Connecting Low-Loss Subspace for Personalized Federated Learning` by **Seok-Ju (Adam) Hahn**, Minwoo Jeong and Junghye lee.  
[ [PAPER](https://arxiv.org/abs/2109.07628) | [POSTER](https://github.com/vaseline555/SuPerFed/blob/main/assets/SIGKDD2022_SuPerFed_Poster_Seok-Ju%20Hahn.pdf) | [SLIDE](https://github.com/vaseline555/SuPerFed/blob/12947b01af3f118b9ae8543d021ae3d256c2b2e7/assets/SIGKDD2022_SuPerFed_Presentation_Seok-Ju%20Hahn.pdf) | [VIDEO](https://dl.acm.org/action/downloadSupplement?doi=10.1145%2F3534678.3539254&file=KDD22-fp0360..mp4) ]

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subspace-learning-for-personalized-federated/personalized-federated-learning-on-mnist-1)](https://paperswithcode.com/sota/personalized-federated-learning-on-mnist-1?p=subspace-learning-for-personalized-federated) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subspace-learning-for-personalized-federated/personalized-federated-learning-on-cifar-10)](https://paperswithcode.com/sota/personalized-federated-learning-on-cifar-10?p=subspace-learning-for-personalized-federated) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subspace-learning-for-personalized-federated/personalized-federated-learning-on-cifar-100)](https://paperswithcode.com/sota/personalized-federated-learning-on-cifar-100?p=subspace-learning-for-personalized-federated) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subspace-learning-for-personalized-federated/personalized-federated-learning-on-tiny)](https://paperswithcode.com/sota/personalized-federated-learning-on-tiny?p=subspace-learning-for-personalized-federated) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subspace-learning-for-personalized-federated/personalized-federated-learning-on-femnist)](https://paperswithcode.com/sota/personalized-federated-learning-on-femnist?p=subspace-learning-for-personalized-federated) \
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/subspace-learning-for-personalized-federated/personalized-federated-learning-on)](https://paperswithcode.com/sota/personalized-federated-learning-on?p=subspace-learning-for-personalized-federated)

# Abstract
Due to the curse of statistical heterogeneity across clients, adopting a personalized federated learning method has become an essential choice for the successful deployment of federated learning-based services. Among diverse branches of personalization techniques, a model mixture-based personalization method is preferred as each client has their own personalized model as a result of federated learning. It usually requires a local model and a federated model, but this approach is either limited to partial parameter exchange or requires additional local updates, each of which is helpless to novel clients and burdensome to the client's computational capacity. As the existence of a connected subspace containing diverse low-loss solutions between two or more independent deep networks has been discovered, we combined this interesting property with the model mixture-based personalized federated learning method for improved performance of personalization.
We proposed SuPerFed, a personalized federated learning method that induces an explicit connection between the optima of the local and the federated model in weight space for boosting each other. Through extensive experiments on several benchmark datasets, we demonstrated that our method achieves consistent gains in both personalization performance and robustness to problematic scenarios possible in realistic services.

# Dependencies
* Please install the required packages first by executing a command `pip install -r requirements.txt`.
```
torch >= 1.9
torchvision >= 0.10
numpy >= 1.20
tensorboard >= 2.8
matplotlib >= 3.5
scipy >= 1.20
```

# Commands
* For detailed description of command arguments, please check `main.py` or run command `python main.py -h`.
## Example command of [Pathological non-IID setting (McMahan et al., 2016)](https://arxiv.org/abs/1602.05629)
```
python3 main.py --exp_name superfed-mm_mnist_patho_50 \
--dataset MNIST --is_small --in_channels 1 --num_classes 10 \
--algorithm superfed-mm --model_name TwoNN --fc_type LinesLinear \
--C 0.1 --K 50 --R 500 --E 10 --B 10 --L 0.6 --nu 1 --mu 0.01 \
--split_type pathological --shard_size 600 \
--eval_every 50
```

## Example command of [Drichlet distribution-based non-IID setting (Hsu et al., 2019)](https://arxiv.org/abs/1909.06335)
```
python3 main.py --exp_name superfed-lm_cifar100_diri_1 \
--dataset CIFAR100 --is_small --in_channels 3 --num_classes 100 \
--algorithm superfed-lm --model_name ResNet9 --fc_type LinesLinear --conv_type LinesConv --bn_type LinesBN \
--C 0.05 --K 100 --R 500 --E 5 --B 20 --L 0.6 --nu 2 --mu 0.01 --lr 0.01 \
--split_type dirichlet --alpha 1.0 --n_jobs 20 \
--eval_every 50
```

## Example command of [Realistic non-IID setting (Caldas et al., 2018; LEAF benchmark](https://leaf.cmu.edu))
```
python3 main.py --exp_name superfed-mm_shakespeare \
--dataset Shakespeare \
--algorithm superfed-mm --model_name NextCharLM --fc_type LinesLinear --lstm_type LinesLSTM --embedding_type LinesEmbedding \
--C 0.008 --R 500 --E 5 --B 50 --lr 0.8 --nu 1.5 --mu 0.01 --L 0.45 --n_jobs 10 --lr_decay 0.995 \
--split_type realistic \
--eval_every 50
```

# Cite in BibTeX
```
@inproceedings{SuPerFed,
author = {Hahn, Seok-Ju and Jeong, Minwoo and Lee, Junghye},
title = {Connecting Low-Loss Subspace for Personalized Federated Learning},
year = {2022},
isbn = {9781450393850},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3534678.3539254},
booktitle = {Proceedings of the 28th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
pages = {505–515},
numpages = {11},
keywords = {label noise, federated learning, personalization, personalized federated learning, non-iid data, mode connectivity},
location = {Washington DC, USA},
series = {KDD '22},
archivePrefix = {arXiv},
arxivId = {2109.07628}
```

# Contact
Feel free to contact the first author (Adam; seokjuhahn@unist.ac.kr) or leave an issue if face with a problem when using this implementation. Thank you! :smiley:
