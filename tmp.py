import torch
from torch.utils.data import DataLoader

from model.encoder import Encoder
from model.decoder import Decoder
from datautil.waymo_dataset import WaymoDataset, waymo_collate_fn

dataset = WaymoDataset('./data/tfrecords', './data/idxs')
dataloader = DataLoader(dataset, batch_size=1, collate_fn=lambda x: waymo_collate_fn(x))

data0 = next(iter(dataloader))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)

states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = data0

states_batch, agents_batch_mask, states_padding_mask_batch, \
                (states_hidden_mask_BP, states_hidden_mask_CBP, states_hidden_mask_GDP), \
                    roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch, \
                        agent_rg_mask, agent_traffic_mask = states_batch.to(device), agents_batch_mask.to(device), states_padding_mask_batch.to(device), \
                                                                        (states_hidden_mask_BP.to(device), states_hidden_mask_CBP.to(device), states_hidden_mask_GDP.to(device)), \
                                                                            roadgraph_feat_batch.to(device), roadgraph_valid_batch.to(device), traffic_light_feat_batch.to(device), traffic_light_valid_batch.to(device), \
                                                                                agent_rg_mask.to(device), agent_traffic_mask.to(device)

encoder = Encoder(device, in_feat_dim=states_batch.shape[-1], in_dynamic_rg_dim=traffic_light_feat_batch.shape[-1], in_static_rg_dim=roadgraph_feat_batch.shape[-1])
encoder = encoder.to(device)

decoder = Decoder(device)
decoder = decoder.to(device)

# TODO : randomly select hidden mask
states_hidden_mask_batch = states_hidden_mask_BP

no_nonpad_mask = torch.sum((~states_padding_mask_batch*~states_hidden_mask_batch),dim=-1) != 0

states_batch = states_batch[no_nonpad_mask]
agents_batch_mask = agents_batch_mask[no_nonpad_mask][:,no_nonpad_mask]
states_padding_mask_batch = states_padding_mask_batch[no_nonpad_mask]
states_hidden_mask_batch = states_hidden_mask_batch[no_nonpad_mask]
agent_rg_mask = agent_rg_mask[no_nonpad_mask]
agent_traffic_mask = agent_traffic_mask[no_nonpad_mask]

roadgraph_valid_mask = roadgraph_valid_batch.sum(dim=-1)!=91
roadgraph_feat_batch = roadgraph_feat_batch[roadgraph_valid_mask]
roadgraph_valid_batch = roadgraph_valid_batch[roadgraph_valid_mask]
agent_rg_mask = agent_rg_mask[:,roadgraph_valid_mask]

traffic_light_valid_mask = traffic_light_valid_batch.sum(dim=-1)!=91
traffic_light_feat_batch = traffic_light_feat_batch[traffic_light_valid_mask]
traffic_light_valid_batch = traffic_light_valid_batch[traffic_light_valid_mask]
agent_traffic_mask = agent_traffic_mask[:,traffic_light_valid_mask]

encodings = encoder(states_batch, agents_batch_mask, states_padding_mask_batch, states_hidden_mask_batch, 
                        roadgraph_feat_batch, roadgraph_valid_batch, traffic_light_feat_batch, traffic_light_valid_batch,
                            agent_rg_mask, agent_traffic_mask)

print(encodings)

# decoding = decoder(encodings, agents_batch_mask, states_padding_mask_batch, 
#                         states_hidden_mask_batch)

# print(decoding.shape)

# to_predict_mask = states_padding_mask_batch*states_hidden_mask_batch
# gt = states_batch[:,:,:6][to_predict_mask] # 6 channel output : x, y, bbox_yaw, velocity_x, velocity_y, vel_yaw

# prediction = decoding.permute(1,2,0,3)[to_predict_mask]

# # print(prediction)

# def some_loss_function(*args):
#     return 0

# loss = some_loss_function(gt, prediction)

# # TODO : training code