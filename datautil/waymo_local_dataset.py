import math
import os
import uuid
import time

#from matplotlib import cm
#import matplotlib.animation as animation
#import matplotlib.pyplot as plt

import numpy as np
#import cv2
# from IPython.display import HTML
import itertools
import torch
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

def waymo_local_collate_fn(batch, halfwidth=100, time_steps=10, GD=16, GS=1400): # GS = max number of static roadgraph element (1400), GD = max number of dynamic roadgraph (16)
    states_batch = np.array([]).reshape(-1,time_steps,9)

    states_padding_mask_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_mask_BP_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_mask_CBP_batch = np.array([]).reshape(-1,time_steps)
    states_hidden_mask_GDP_batch =np.array([]).reshape(-1,time_steps)

    roadgraph_feat_batch = np.array([]).reshape(-1,time_steps,6)
    roadgraph_padding_batch = np.array([]).reshape(-1,time_steps)

    traffic_light_feat_batch = np.array([]).reshape(-1,time_steps,3)
    traffic_light_padding_batch = np.array([]).reshape(-1,time_steps)

    num_agents = np.array([])
    num_rg = np.array([])
    num_tl = np.array([])

    for data in batch:
        # State of Agents
        past_states = np.stack((data['state/past/x'],data['state/past/y'],data['state/past/bbox_yaw'],
                                    data['state/past/velocity_x'],data['state/past/velocity_y'],data['state/past/vel_yaw'],
                                        data['state/past/width'],data['state/past/length'],data['state/past/timestamp_micros']), axis=-1)
        past_states_valid = data['state/past/valid'] > 0.
        current_states = np.stack((data['state/current/x'],data['state/current/y'],data['state/current/bbox_yaw'],
                                    data['state/current/velocity_x'],data['state/current/velocity_y'],data['state/current/vel_yaw'],
                                        data['state/current/width'],data['state/current/length'],data['state/current/timestamp_micros']), axis=-1)
        current_states_valid = data['state/current/valid'] > 0.
        future_states = np.stack((data['state/future/x'],data['state/future/y'],data['state/future/bbox_yaw'],
                                    data['state/future/velocity_x'],data['state/future/velocity_y'],data['state/future/vel_yaw'],
                                        data['state/future/width'],data['state/future/length'],data['state/future/timestamp_micros']), axis=-1)
        future_states_valid = data['state/future/valid'] > 0.

        sdc_mask = data['state/is_sdc'] > 0.
        agent_types = data['state/type']

        states_feat = np.concatenate((past_states,current_states,future_states),axis=1)                             # [A,T,D]
        states_padding_mask = ~np.concatenate((past_states_valid,current_states_valid,future_states_valid), axis=1) # [A,T]

        states_feat = states_feat[:,::5,:][:,:10,:]
        states_padding_mask = states_padding_mask[:,::5][:,:10]

        center_x, center_y = states_feat[sdc_mask][0,3,:2]
        
        # make global coordinate to local centered to sdv and normalize 
        # Also, pad features outside width. Also, update padding mask 
        states_feat[:,:,:2] -= np.array([center_x,center_y])
        agent_xy_mask = (states_feat[:,:,:2] >= -halfwidth) * (states_feat[:,:,:2] <= halfwidth)
        agent_xy_mask = agent_xy_mask[:,:,0]*agent_xy_mask[:,:,1]
        states_feat[:,:,:2] /= halfwidth
        states_feat[~agent_xy_mask] = -1
        states_padding_mask += ~agent_xy_mask

        # filter out if agent is unvalid across all time steps
        states_any_mask = np.sum(states_padding_mask,axis=1) != time_steps
        states_feat = states_feat[states_any_mask]
        states_padding_mask = states_padding_mask[states_any_mask]

        # Mask only vehicles
        agent_type_mask = agent_types[states_any_mask]==1
        states_feat = states_feat[agent_type_mask]
        states_padding_mask = states_padding_mask[agent_type_mask]

        num_agents = np.append(num_agents, len(states_feat))

        # basic_mask = np.zeros((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_mask_BP = np.ones((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_mask_BP[:,:4] = False
        sdvidx = np.where(data['state/is_sdc'][states_any_mask][agent_type_mask] == 1)[0][0]
        states_hidden_mask_CBP = np.ones((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_mask_CBP[:,:4] = False
        states_hidden_mask_CBP[sdvidx,:] = False
        states_hidden_mask_GDP = np.ones((len(states_feat),time_steps)).astype(np.bool_)
        states_hidden_mask_GDP[:,:4] = False
        states_hidden_mask_GDP[sdvidx,-1] = False
        # states_hidden_mask_CDP = np.zeros((len(states_feat),time_steps)).astype(np.bool_)
        
        # Static Road Graph
        roadgraph_feat = np.concatenate((data['roadgraph_samples/id'], data['roadgraph_samples/type'], 
                                            data['roadgraph_samples/xyz'][:,:2], data['roadgraph_samples/dir'][:,:2]), axis=-1)
        roadgraph_valid = data['roadgraph_samples/valid'] > 0.

        roadgraph_feat = roadgraph_feat[roadgraph_valid[:,0]]
        roadgraph_feat[:,2:4] -= np.array([center_x,center_y])
        rg_xy_mask = (roadgraph_feat[:,2:4] >= -halfwidth) * (roadgraph_feat[:,2:4] <= halfwidth)
        rg_xy_mask = rg_xy_mask[:,0]*rg_xy_mask[:,1]
        roadgraph_feat[:,2:4] /= halfwidth
        roadgraph_feat = roadgraph_feat[rg_xy_mask]
        roadgraph_valid = np.ones((roadgraph_feat.shape[0],1)).astype(np.bool_)

        if roadgraph_feat.shape[0] > GS:
            spacing = roadgraph_feat.shape[0] // GS
            roadgraph_feat = roadgraph_feat[::spacing, :]
            remove_num = len(roadgraph_feat) - GS
            roadgraph_mask2 = np.full(len(roadgraph_feat), True)
            idx_remove = np.random.choice(range(len(roadgraph_feat)), remove_num, replace=False)
            roadgraph_mask2[idx_remove] = False
            roadgraph_feat = roadgraph_feat[roadgraph_mask2]
            roadgraph_valid = np.ones((GS,1)).astype(np.bool_)
            num_rg = np.append(num_rg, GS)
        else:
            num_rg = np.append(num_rg, roadgraph_feat.shape[0])

        roadgraph_feat = np.repeat(roadgraph_feat[:,np.newaxis,:],time_steps,axis=1)
        roadgraph_valid = np.repeat(roadgraph_valid,time_steps,axis=1)
        roadgraph_padding = ~roadgraph_valid

        # Dynamic Road Graph
        traffic_light_states_past = np.stack((data['traffic_light_state/past/state'].T,data['traffic_light_state/past/x'].T,data['traffic_light_state/past/y'].T),axis=-1)
        traffic_light_valid_past = data['traffic_light_state/past/valid'].T > 0.
        traffic_light_states_current = np.stack((data['traffic_light_state/current/state'].T,data['traffic_light_state/current/x'].T,data['traffic_light_state/current/y'].T),axis=-1)
        traffic_light_valid_current = data['traffic_light_state/current/valid'].T > 0.
        traffic_light_states_future = np.stack((data['traffic_light_state/future/state'].T,data['traffic_light_state/future/x'].T,data['traffic_light_state/future/y'].T),axis=-1)
        traffic_light_valid_future = data['traffic_light_state/future/valid'].T > 0.

        traffic_light_feat = np.concatenate((traffic_light_states_past,traffic_light_states_current,traffic_light_states_future),axis=1)
        traffic_light_valid = np.concatenate((traffic_light_valid_past,traffic_light_valid_current,traffic_light_valid_future),axis=1)
        traffic_light_feat = traffic_light_feat[:,::5,:][:,:10,:]
        traffic_light_valid = traffic_light_valid[:,::5][:,:10]
        traffic_light_padding = ~traffic_light_valid

        traffic_light_feat[:,:,1:3] -= np.array([center_x,center_y])
        tl_xy_mask = (traffic_light_feat[:,:,1:3] >= -halfwidth) * (traffic_light_feat[:,:,1:3] <= halfwidth)
        tl_xy_mask = tl_xy_mask[:,:,0]*tl_xy_mask[:,:,1]
        traffic_light_feat[:,:,1:3] /= halfwidth
        traffic_light_feat[~tl_xy_mask] = -1
        traffic_light_padding += ~tl_xy_mask

        tl_any_mask = traffic_light_padding.sum(axis=-1) != time_steps
        traffic_light_feat = traffic_light_feat[tl_any_mask]
        traffic_light_padding = traffic_light_padding[tl_any_mask]
        num_tl = np.append(num_tl, traffic_light_feat.shape[0])

        # Concat across batch
        states_batch = np.concatenate((states_batch,states_feat), axis=0)
        states_padding_mask_batch = np.concatenate((states_padding_mask_batch,states_padding_mask), axis=0)

        states_hidden_mask_BP_batch = np.concatenate((states_hidden_mask_BP_batch,states_hidden_mask_BP), axis=0)
        states_hidden_mask_CBP_batch = np.concatenate((states_hidden_mask_CBP_batch,states_hidden_mask_CBP), axis=0)
        states_hidden_mask_GDP_batch =np.concatenate((states_hidden_mask_GDP_batch,states_hidden_mask_GDP), axis=0)

        roadgraph_feat_batch = np.concatenate((roadgraph_feat_batch, roadgraph_feat), axis=0)
        roadgraph_padding_batch = np.concatenate((roadgraph_padding_batch, roadgraph_padding), axis=0)

        traffic_light_feat_batch = np.concatenate((traffic_light_feat_batch, traffic_light_feat), axis=0)
        traffic_light_padding_batch = np.concatenate((traffic_light_padding_batch, traffic_light_padding), axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int32)
    num_rg_accum = np.cumsum(np.insert(num_rg,0,0)).astype(np.int32)
    num_tl_accum = np.cumsum(np.insert(num_tl,0,0)).astype(np.int32)

    agents_batch_mask = np.ones((num_agents_accum[-1],num_agents_accum[-1])) # padding. 1 -> padded (ignore att) / 0 -> non-padded (do att)
    agent_rg_mask = np.ones((num_agents_accum[-1],num_rg_accum[-1]))
    agent_traffic_mask = np.ones((num_agents_accum[-1],num_tl_accum[-1]))

    for i in range(len(num_agents)):
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 0
        agent_rg_mask[num_agents_accum[i]:num_agents_accum[i+1], num_rg_accum[i]:num_rg_accum[i+1]] = 0
        agent_traffic_mask[num_agents_accum[i]:num_agents_accum[i+1], num_tl_accum[i]:num_tl_accum[i+1]] = 0

    states_batch = torch.FloatTensor(states_batch)
    agents_batch_mask = torch.BoolTensor(agents_batch_mask)
    states_padding_mask_batch = torch.BoolTensor(states_padding_mask_batch)
    states_hidden_mask_BP_batch = torch.BoolTensor(states_hidden_mask_BP_batch)
    states_hidden_mask_CBP_batch = torch.BoolTensor(states_hidden_mask_CBP_batch)
    states_hidden_mask_GDP_batch = torch.BoolTensor(states_hidden_mask_GDP_batch)
    
    roadgraph_feat_batch = torch.FloatTensor(roadgraph_feat_batch)
    roadgraph_padding_batch = torch.BoolTensor(roadgraph_padding_batch)
    traffic_light_feat_batch = torch.FloatTensor(traffic_light_feat_batch)
    traffic_light_padding_batch = torch.BoolTensor(traffic_light_padding_batch)

    agent_rg_mask = torch.BoolTensor(agent_rg_mask)
    agent_traffic_mask = torch.BoolTensor(agent_traffic_mask)   
        
    return (states_batch, agents_batch_mask, states_padding_mask_batch, 
                (states_hidden_mask_BP_batch, states_hidden_mask_CBP_batch, states_hidden_mask_GDP_batch), 
                    roadgraph_feat_batch, roadgraph_padding_batch, traffic_light_feat_batch, traffic_light_padding_batch,
                        agent_rg_mask, agent_traffic_mask)
