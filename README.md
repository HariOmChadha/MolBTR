### Updated Approach: Molecular Barlow Twin Learning of Representations via Graph Neural Networks ##

<img src="figs/BT_pipeline.png" width="450">

The original implementation has been updated to use the augmentations for the molecules to train two models simulaneously using a Barlow Twin approach: [https://arxiv.org/abs/2103.03230]. The models can then be finetuned to predict the viscosity/thermal conductivity of thermal fluids. 

## Getting Started

### Installation

Set up conda environment and clone the github repo

```
# create a new environment
$ conda create --name molbtr python=3.11
$ conda activate molbtr

# install requirements
$ pip install -r requirements.txt

# clone the source code of MolCLR
$ git clone https://github.com/HariOmChadha/MolBTR.git
$ cd MolBTR
```

### Dataset

The datasets used can be found here: 

### Pre-training

To train the MolBTR, where the configurations and detailed explaination for each variable can be found in `config.yaml`
```
Use the jupyter notebook called main.ipynb
```
To monitor the training via tensorboard, run `tensorboard --logdir ckpt/{PATH}` and click the URL http://127.0.0.1:6006/.

### Fine-tuning 

To fine-tune the MolBTR pre-trained model on downstream molecular benchmarks, where the configurations and detailed explaination for each variable can be found in `config_finetune.yaml`. IMPORTANT: Use the datasets in the format provided. Currently, can only finetune for Dynamic Viscosity and Thermal Conductivity. 
```
Use the jupyter notebook called main.ipynb 
```

### Pre-trained models

We also provide pre-trained GCN models, which can be found in `ckpt/BT1` and `ckpt/BT2` respectively.
BT1: 15% subgraph removed
BT2: 15% subgraph removed + 20% nodes/egdes masked

## Acknowledgements
- MolCLR Approach: [https://www.nature.com/articles/s42256-022-00447-x#code-availability]
- Barlow Twin Approach: [https://arxiv.org/abs/2103.03230]
- PyTorch implementation of SimCLR: [https://github.com/sthalles/SimCLR](https://github.com/sthalles/SimCLR)
- Strategies for Pre-training Graph Neural Networks: [https://github.com/snap-stanford/pretrain-gnns](https://github.com/snap-stanford/pretrain-gnns)
