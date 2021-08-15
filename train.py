import os
import sys
import hydra

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
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
    dataset_valid = WaymoDataset(pwd+cfg.dataset.valid.tfrecords, pwd+cfg.dataset.valid.idxs)
    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.batchsize, collate_fn=waymo_collate_fn)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.batchsize, collate_fn=waymo_collate_fn)

    model = SceneTransformer(None, cfg.model.in_feature_dim, cfg.model.in_dynamic_rg_dim, cfg.model.in_static_rg_dim,
                                cfg.model.time_steps, cfg.model.feature_dim, cfg.model.head_num, cfg.model.k, cfg.model.F)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num)
    trainer.fit(model, dloader_train, dloader_valid)

if __name__ == '__main__':
    sys.exit(main())
