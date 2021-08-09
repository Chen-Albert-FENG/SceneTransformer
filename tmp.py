import torch
from torch.utils.data import DataLoader

from model.encoder import Encoder
from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn

dataset = WaymoDataset('data')
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: waymo_collate_fn(x))

data0 = next(iter(dataloader))

device = torch.cuda.device(0)

print(device)

state_feat, agent_batch_mask, road_feat, traffic_light_feat = data0
# state_feat, agent_batch_mask, road_feat, traffic_light_feat = state_feat.to(device), \
#                                                                     agent_batch_mask.to(device), \
#                                                                         road_feat.to(device), \
#                                                                             traffic_light_feat.to(device)

encoder = Encoder(in_feat_dim=state_feat.shape[-1], in_dynamic_rg_dim=traffic_light_feat.shape[-1], in_static_rg_dim=road_feat.shape[-1])
# encoder = encoder.to(device)

encodings = encoder(state_feat, agent_batch_mask, road_feat, traffic_light_feat)

print(encodings.shape)