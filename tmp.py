import torch
from torch.utils.data import DataLoader

from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn

dataset = WaymoDataset('data')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=lambda x: waymo_collate_fn(x))

data0 = next(iter(dataloader))