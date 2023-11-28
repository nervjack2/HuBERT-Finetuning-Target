import sys 
import os 
from copy import deepcopy
import torchaudio
import torch
import argparse
from torch.utils.data import Dataset, DataLoader 
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from fairseq.checkpoint_utils import load_checkpoint_to_cpu
from fairseq.tasks.hubert_pretraining import HubertPretrainingConfig
from fairseq.data.dictionary import Dictionary
from omegaconf import OmegaConf
from dataclasses import dataclass, is_dataclass
from tqdm import tqdm

class InputDataset(Dataset):
    def __init__(self, tsv_pth, task_cfg):
        super(InputDataset, self).__init__()
        self.file_pth = self.get_wav_path(tsv_pth)
        self.task_cfg = task_cfg

    def get_wav_path(self, tsv_path):
        # Open tsv file 
        file_pth = []
        with open(tsv_path, 'r') as fp:
            root_path = fp.readline().strip()
            for x in fp:
                info = x.strip().split('\t')
                file_pth.append(info[0])
        file_pth = [os.path.join(root_path, x) for x in file_pth] 
        return file_pth
    
    def read_audio(self, path):
        wav, sr = torchaudio.load(path)
        assert sr == 16000, sr
        return wav   

    def pth2name(self, path):
        return path.split('/')[-1].split('.')[0]

    def __len__(self):
        return len(self.file_pth)

    def __getitem__(self, idx):
        return self.read_audio(self.file_pth[idx])

    def collate_fn(self, items):
        wavs = [wav.squeeze(0) for wav in items]
        if self.task_cfg.normalize:
            wavs = [F.layer_norm(wav, wav.shape) for wav in wavs]

        wav_lengths = torch.LongTensor([len(wav) for wav in wavs])
        wav_padding_mask = ~torch.lt(
            torch.arange(max(wav_lengths)).unsqueeze(0),
            wav_lengths.unsqueeze(1),
        )
    
        padded_wav = pad_sequence(wavs, batch_first=True)

        return padded_wav, wav_padding_mask

def merge_with_parent(dc: dataclass, cfg: dict):
    # Reference: S3PRL toolkit (https://github.com/s3prl/s3prl/blob/main/s3prl/upstream/utils.py#L31)
    # Reference Author: Shu-wen Yang (https://github.com/leo19941227)
    assert is_dataclass(dc)
    assert type(cfg) == dict
    cfg = deepcopy(cfg)

    def fix_cfg(cfg):
        target_keys = set(dc.__dataclass_fields__.keys())
        for k in list(cfg.keys()):
            if k not in target_keys:
                del cfg[k]
    fix_cfg(cfg)
    assert len(cfg) > 0
    return dc(**cfg)


def main(tsv_pth, model_pth, save_dir, batch_size=1, disable_dropout=True):
    os.makedirs(save_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load pretrained HuBERT 
    state = load_checkpoint_to_cpu(model_pth)
    cfg = OmegaConf.to_container(state["cfg"])
    from hubert_model import HubertModel, HubertConfig
    model_cfg = merge_with_parent(HubertConfig, cfg["model"])
    task_cfg = merge_with_parent(HubertPretrainingConfig, cfg["task"])
    dicts: List[Dictionary] = state["task_state"]["dictionaries"]
    num_tar = len(dicts)
    print('Number of targets:', num_tar)
    symbols = [dictionary.symbols for dictionary in dicts]

    # Disable dropout
    if disable_dropout:
        model_cfg.dropout = 0.0
        model_cfg.attention_dropout = 0.0
        model_cfg.activation_dropout = 0.0 
        model_cfg.encoder_layerdrop = 0.0
        model_cfg.dropout_input = 0.0
        model_cfg.dropout_features = 0.0

    model = HubertModel(model_cfg, task_cfg, symbols).to(device)
    model.load_state_dict(state["model"])
    model.train()
    # Load data iterator
    dataset = InputDataset(tsv_pth, task_cfg)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=dataset.collate_fn)

    # Write result in to a file 
    tsv_name = tsv_pth.split('/')[-1].split('.')[0]
    fp = [open(f'{save_dir}/{tsv_name}-{i}.km', 'w') for i in range(num_tar)]

    for wavs, padding_mask in tqdm(loader):
        wavs = wavs.to(device)
        padding_mask = padding_mask.to(device)
        with torch.no_grad():
            state = model(source=wavs, padding_mask=padding_mask, mask=False)
        padding_mask = state["padding_mask"]
        logit_list = state["logit"]
        for i, logit in enumerate(logit_list):
            feat_len = padding_mask.sum(dim=-1).cpu().numpy()
            target = torch.argmax(logit, dim=-1).cpu().numpy()
            cur_idx = 0
            for _, l in enumerate(feat_len):
                t = ' '.join([str(x) for x in target[cur_idx:cur_idx+l].tolist()])
                cur_idx += l 
                fp[i].write(f'{t}\n')

    for i in range(num_tar):
        fp[i].close()        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--tsv-pth', type=str, help="fairseq .tsv file")
    parser.add_argument('-m', '--model-pth', type=str, help="pretrained HuBERT model checkpoint")
    parser.add_argument('-s', '--save-dir', default='./label' , type=str, help="directory to save generated label")
    parser.add_argument('-b', '--batch-size', default=1, type=int, help="batch size")
    parser.add_argument('-d', '--disable-dropout', default=True, type=bool, help="whether to disable dropout")
    args = parser.parse_args()
    main(**vars(args))
