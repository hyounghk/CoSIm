# CoSIm
Code and dataset for NAACL 2022 paper "CoSIm: Commonsense Reasoning for Counterfactual Scene Imagination" Hyounghun Kim, Abhay Zala, Mohit Bansal.


## Prerequisites

- Python 3.8
- [PyTorch 1.4](http://pytorch.org/) or Up
- For others packages, please run this command.
```
pip install -r requirements.txt
```

## Dataset
Please download image features etc. from [here](https://drive.google.com/file/d/1hm-CKavVeKt0ixzKRfyxSrBqeey0tqRz/view?usp=sharing) and unzip in data/cosim_feats folder.<br>

## Usage

To train the models, please run the script run/lxmert_pretrain.bash like below:
```
bash run/run_train.bash 0  
```
