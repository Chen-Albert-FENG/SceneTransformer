import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
import cv2
# from IPython.display import HTML
import itertools
import torch
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

def xy_to_pixel(xy, width):
    xy[:,1] *= -1
    xy += int(width/2)
    mask_x = (xy[:,0] <= width)*(xy[:,0]>=0)
    mask_y = (xy[:,1] <= width)*(xy[:,1]>=0)

    xy = xy[mask_x*mask_y]

    return xy

def waymo_raster_collate_fn(batch, GD=16, GS=1400): # GS = max number of static roadgraph element (1400), GD = max number of dynamic roadgraph (16)
    scene_img_batch = np.array([]).reshape(-1,500,500)
    tgt_img_batch = np.array([]).reshape(-1,500,500)
    agent_points_batch = np.array([]).reshape(-1,4)

    states_feat_batch = np.array([]).reshape(-1,91,9)
    states_padding_mask_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_BP_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_CBP_batch = np.array([]).reshape(-1,91)
    states_hidden_mask_GDP_batch =np.array([]).reshape(-1,91)

    num_agents = np.array([])

    width = 150
    empty_mask = np.zeros((width,width))

    for data in batch:
        # State of Agents
        vehicle_mask = (data['state/type']==1)

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

        states_feat = np.concatenate((past_states,current_states,future_states),axis=1)
        states_valid = np.concatenate((past_states_valid,current_states_valid,future_states_valid),axis=1)
        states_any_mask = np.sum(states_valid,axis=1) > 0

        states_feat = states_feat[states_any_mask*vehicle_mask]
        states_padding_mask = states_valid[states_any_mask*vehicle_mask]
        
        # basic_mask = np.zeros((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_BP = np.ones((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_BP[:,:12] = False
        sdvidx = np.where(data['state/is_sdc'][states_any_mask] == 1)[0][0]
        states_hidden_mask_CBP = np.zeros((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_CBP[:,:12] = False
        states_hidden_mask_CBP[sdvidx-1,:] = False
        states_hidden_mask_GDP = np.zeros((len(states_feat),91)).astype(np.bool_)
        states_hidden_mask_GDP[:,:12] = False
        states_hidden_mask_GDP[sdvidx-1,-1] = False
        # states_hidden_mask_CDP = np.zeros((len(states_feat),91)).astype(np.bool_)


        roadgraph_xyz = data['roadgraph_samples/xyz']
        roadgraph_type = data['roadgraph_samples/type']
        roadgraph_id = data['roadgraph_samples/id']

        # Dynamic Road Graph
        traffic_light_states_current = np.stack((data['traffic_light_state/current/state'].T,data['traffic_light_state/current/x'].T,data['traffic_light_state/current/y'].T),axis=-1)
        traffic_light_valid_current = data['traffic_light_state/current/valid'].T > 0.

        '''
        SDV-Centered Rasterized Images : [6,500,500]
        
        6-Channel Input : 
            0: drivable area
            1: centerlines
            2: road line
            3: red lights
            4: yellow lights
            5: green lights 
        
        (Optional) : add 3 more channel -> arrow red light, arrow yellow light, arrow green light

        '''
        sdc_mask = data['state/is_sdc'] > 0.
        center_x, center_y = current_states[sdc_mask][0,0,:2]

        ctline_mask = (roadgraph_type==2)[:,0]

        drivable_area = empty_mask.copy()
        centerlines = empty_mask.copy()

        for id_ in np.unique(roadgraph_id[ctline_mask]):
            ctline_id_mask = ctline_mask*(roadgraph_id==id_)[:,0]
            ctline_xy = roadgraph_xyz[ctline_id_mask][:,:2] - np.array([center_x,center_y])
            ctline_xy = xy_to_pixel(ctline_xy,width)

            polygon = np.array([ctline_xy], np.int32)
            cv2.polylines(drivable_area, polygon, isClosed=False, color=1, thickness=5)
            cv2.polylines(centerlines, polygon, isClosed=False, color=1, thickness=1)

        # Road 
        lane_mask = np.zeros(ctline_mask.shape).astype(np.bool_)
        for type_ in [6,7,8,9,10,11,12,13,14]:
            lane_mask += (roadgraph_type==type_)[:,0]

        lanes = empty_mask.copy()

        for id_ in np.unique(roadgraph_id[lane_mask]):
            lane_id_mask = lane_mask*(roadgraph_id==id_)[:,0]
            lane_xy = roadgraph_xyz[lane_id_mask][:,:2] - np.array([center_x,center_y])
            lane_xy = xy_to_pixel(lane_xy,width)

            polygon = np.array([lane_xy], np.int32)
            cv2.polylines(lanes, polygon, isClosed=False, color=1, thickness=1)


        red_mask = ((traffic_light_states_current[:,0,0] == 4) + (traffic_light_states_current[:,0,0] == 7))
        yellow_mask = ((traffic_light_states_current[:,0,0] == 5) + (traffic_light_states_current[:,0,0] == 8))
        green_mask = traffic_light_states_current[:,0,0] == 6

        # arrow_red_mask = traffic_light_states_current[:,0,0] == 1
        # arrow_yellow_mask = traffic_light_states_current[:,0,0] == 2
        # arrow_green_mask = traffic_light_states_current[:,0,0] == 3

        red_xy = xy_to_pixel(traffic_light_states_current[:,0,1:][red_mask] - np.array([center_x,center_y]), width)
        yellow_xy = xy_to_pixel(traffic_light_states_current[:,0,1:][yellow_mask] - np.array([center_x,center_y]), width)
        green_xy = xy_to_pixel(traffic_light_states_current[:,0,1:][green_mask] - np.array([center_x,center_y]), width)

        red_lights = empty_mask.copy()
        yellow_lights = empty_mask.copy()
        green_lights = empty_mask.copy()

        [cv2.circle(img=red_lights, center=tuple(xy.astype(np.int32)), radius=1, color=1, thickness=cv2.FILLED) for xy in red_xy]
        [cv2.circle(img=yellow_lights, center=tuple(xy.astype(np.int32)), radius=1, color=1, thickness=cv2.FILLED) for xy in yellow_xy]
        [cv2.circle(img=green_lights, center=tuple(xy.astype(np.int32)), radius=1, color=1, thickness=cv2.FILLED) for xy in green_xy]

        scene_img = np.stack((drivable_area,centerlines,lanes,red_lights,yellow_lights,green_lights))

        # filter out vehicle located outside boundary 
        states_feat[:,:,:2] = states_feat[:,:,:2] - np.array([center_x,center_y])
        agent_xy_mask = (-width/2 <= states_feat[:,10,:2])*(states_feat[:,10,:2] <= width/2)
        agent_xy_mask = agent_xy_mask[:,0]*agent_xy_mask[:,1]
        states_feat = states_feat[agent_xy_mask]

        states_padding_mask = states_padding_mask[agent_xy_mask]
        states_hidden_mask_BP = states_hidden_mask_BP[agent_xy_mask]
        states_hidden_mask_CBP = states_hidden_mask_CBP[agent_xy_mask]
        states_hidden_mask_GDP = states_hidden_mask_GDP[agent_xy_mask]

        # make target occupancy map for each agent
        tgt_img = np.zeros((len(states_feat),width,width))
        for i, agent_xy in enumerate(states_feat[:,10:,:2]):
            agent_xy = xy_to_pixel(agent_xy,width)
            cv2.polylines(tgt_img[i],[agent_xy.astype(np.int32)],isClosed=False,color=1,thickness=1)

        # normalize coordinate
        states_feat[:,10,:2] /= (width/2)       # normalize to [-1,1]
        # agent start point and end point 
        # 4-channel : start_x, start_y, end_x, end_y
        agent_points = np.concatenate((states_feat[:,10,:2],states_feat[:,-1,:2]),axis=-1)

        # resize to [500,500]
        scene_img = cv2.resize(scene_img.transpose(1,2,0),(500,500)).transpose(2,0,1)
        tgt_img = cv2.resize(tgt_img.transpose(1,2,0),(500,500)).transpose(2,0,1)

        num_agents = np.append(num_agents, len(states_feat))
    

        scene_img_batch = np.concatenate((scene_img_batch,scene_img),axis=0)
        tgt_img_batch = np.concatenate((tgt_img_batch,tgt_img),axis=0)
        agent_points_batch = np.concatenate((agent_points_batch,agent_points),axis=0)
        states_feat_batch = np.concatenate((states_feat_batch,states_feat),axis=0)
        states_padding_mask_batch = np.concatenate((states_padding_mask_batch,states_padding_mask),axis=0)
        states_hidden_mask_BP_batch = np.concatenate((states_hidden_mask_BP_batch,states_hidden_mask_BP),axis=0)
        states_hidden_mask_CBP_batch = np.concatenate((states_hidden_mask_CBP_batch,states_hidden_mask_CBP),axis=0)
        states_hidden_mask_GDP_batch = np.concatenate((states_hidden_mask_GDP_batch,states_hidden_mask_GDP),axis=0)

    num_agents_accum = np.cumsum(np.insert(num_agents,0,0)).astype(np.int64)
    agents_batch_mask = np.zeros((num_agents_accum[-1],num_agents_accum[-1]))
    for i in range(len(num_agents)):
        agents_batch_mask[num_agents_accum[i]:num_agents_accum[i+1], num_agents_accum[i]:num_agents_accum[i+1]] = 1

    return (scene_img_batch, tgt_img_batch, agent_points_batch, states_feat_batch, agents_batch_mask,
                states_padding_mask_batch, (states_hidden_mask_BP_batch,states_hidden_mask_CBP_batch,states_hidden_mask_GDP_batch))
