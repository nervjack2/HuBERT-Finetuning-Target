# HuBERT-Finetuning-Target
**This is a tool for generating HuBERT fine-tuning target.** 

## Introduction
Since fairseq does not release the targets for HuBERT, it is hard for community to fine-tune HuBERT.\
This tool provides a simple way to speculating the targets of HuBERT.\
Concretly speaking, we take the class with the highest probability as the pseudo label.
This method is not optimal, since HuBERT must not predict perfectly,\
however, it is a quick and effective way to generate a set of label which is closed to the true one.
 
Note: fairseq do have release a quantizer in [L9 km500](https://github.com/facebookresearch/fairseq/tree/main/examples/hubert). However, this quantizer is actually training by the 9th layer of **HuBERT's second stage model**. The targets produced by this quantizer would not be identical to the ones used in HuBERT's second stage training. (It would be more like a third stage training). 

## Environment
Please Install [fairseq](https://github.com/facebookresearch/fairseq/tree/main)

## Usage 
```
python3 gen_stg2_tgt.py -t [tsv path] -m [pretraind HuBERT model path] -b [batch size]
```

-t: tsv path in fairseq format. See dev-clean.tsv for example\
-m: pretrained HuBERT model path\
-b: batch size\
-s: the directory to save generated lable\
-d: whether to disable dropout in HuBERT 

## Fine-tuning performance 

ASR: 7.06%
SID: 82.01%

- Fine-tuning datast: Librispeech 960 hours
- Pretrained model: HuBERT base second stage
- Fine-tuning batch size: 32
- Fine-tuning learning rate: 5e-5
- Fine-tuing epochs: 10
- Fine-tuning time: 24 hours



