import os
import sys
import hydra

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn
from datautil.waymo_local_dataset import waymo_local_collate_fn
from model.pl_module import SceneTransformer



@hydra.main(config_path='./conf', config_name='config.yaml')
def main(cfg):
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids

    pl.seed_everything(cfg.seed)
    pwd = hydra.utils.get_original_cwd() + '/'
    print('Current Path: ', pwd)

    dataset_train = WaymoDataset(pwd+cfg.dataset.train.tfrecords, pwd+cfg.dataset.train.idxs)
    dataset_valid = WaymoDataset(pwd+cfg.dataset.valid.tfrecords, pwd+cfg.dataset.valid.idxs)
    dloader_train = DataLoader(dataset_train, batch_size=cfg.dataset.batchsize, collate_fn=waymo_local_collate_fn, num_workers=16)
    dloader_valid = DataLoader(dataset_valid, batch_size=cfg.dataset.batchsize, collate_fn=waymo_local_collate_fn, num_workers=16)

    model = SceneTransformer(cfg.model.in_feature_dim, cfg.model.in_dynamic_rg_dim, cfg.model.in_static_rg_dim,
                                cfg.model.time_steps, cfg.model.feature_dim, cfg.model.head_num, cfg.model.k, cfg.model.F)

    trainer = pl.Trainer(max_epochs=cfg.max_epochs, gpus=cfg.device_num, gradient_clip_val=5,accelerator='ddp')
    try:
        trainer.fit(model, dloader_train, dloader_valid)
    except Exception as e:
        print('fit error: ',e)
        import pdb
        pdb.set_trace()

if __name__ == '__main__':
    sys.exit(main())
