import os
import sys
import hydra
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, dataloader
import pytorch_lightning as pl

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
from datautil.waymo_local_dataset import waymo_local_collate_fn
from model.pl_module import SceneTransformer



@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    # GPU_NUM = cfg.device_num
    # device = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)
    # if device.type == 'cuda':
    #     print(torch.cuda.get_device_name(GPU_NUM))
    #     print('Memory Usage:')
    #     print('Allocated:', round(torch.cuda.memory_allocated(GPU_NUM)/1024**3,1), 'GB')
    #     print('Cached:   ', round(torch.cuda.memory_cached(GPU_NUM)/1024**3,1), 'GB')

    pl.seed_everything(cfg.seed)
    pwd = hydra.utils.get_original_cwd() + '/'
    print('Current Path: ', pwd)

    dataset_train = WaymoDataset(pwd+cfg.dataset.train.tfrecords, pwd+cfg.dataset.train.idxs)
    # dataset_valid = WaymoDataset(pwd+cfg.dataset.valid.tfrecords, pwd+cfg.dataset.valid.idxs)
    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.batchsize, collate_fn=waymo_local_collate_fn)
    # dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.batchsize, collate_fn=waymo_local_collate_fn, shuffle=False)
    #print(len(dataset_valid)) 
    for data in tqdm(dloader_train):
        states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = data
        pass

    print('finished')

if __name__ == '__main__':
    sys.exit(main())
