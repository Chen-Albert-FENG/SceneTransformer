import math
import os
import uuid
import time

from matplotlib import cm
import matplotlib.animation as animation
import matplotlib.pyplot as plt

import numpy as np
from IPython.display import HTML
import itertools
import torch
from tfrecord.torch.dataset import TFRecordDataset, MultiTFRecordDataset

from google.protobuf import text_format
from waymo_open_dataset.metrics.ops import py_metrics_ops
from waymo_open_dataset.metrics.python import config_util_py as config_util
from waymo_open_dataset.protos import motion_metrics_pb2

# Example field definition
roadgraph_features = {
    'roadgraph_samples/dir':
        'float',
    'roadgraph_samples/id':
        'int',
    'roadgraph_samples/type':
        'int',
    'roadgraph_samples/valid':
        'int',
    'roadgraph_samples/xyz':
        'float',
}

# Features of other agents.
state_features = {
    'state/id':
        'float',
    'state/type':
        'float',
    'state/is_sdc':
        'int',
    'state/tracks_to_predict':
        'int',
    'state/current/bbox_yaw':
        'float',
    'state/current/height':
        'float',
    'state/current/length':
        'float',
    'state/current/timestamp_micros':
        'int',
    'state/current/valid':
        'int',
    'state/current/vel_yaw':
        'float',
    'state/current/velocity_x':
        'float',
    'state/current/velocity_y':
        'float',
    'state/current/width':
        'float',
    'state/current/x':
        'float',
    'state/current/y':
        'float',
    'state/current/z':
        'float',
    'state/future/bbox_yaw':
        'float',
    'state/future/height':
        'float',
    'state/future/length':
        'float',
    'state/future/timestamp_micros':
        'int',
    'state/future/valid':
        'int',
    'state/future/vel_yaw':
        'float',
    'state/future/velocity_x':
        'float',
    'state/future/velocity_y':
        'float',
    'state/future/width':
        'float',
    'state/future/x':
        'float',
    'state/future/y':
        'float',
    'state/future/z':
        'float',
    'state/past/bbox_yaw':
        'float',
    'state/past/height':
        'float',
    'state/past/length':
        'float',
    'state/past/timestamp_micros':
        'int',
    'state/past/valid':
        'int',
    'state/past/vel_yaw':
        'float',
    'state/past/velocity_x':
        'float',
    'state/past/velocity_y':
        'float',
    'state/past/width':
        'float',
    'state/past/x':
        'float',
    'state/past/y':
        'float',
    'state/past/z':
        'float',
}

traffic_light_features = {
    'traffic_light_state/current/state':
        'int',
    'traffic_light_state/current/valid':
        'int',
    'traffic_light_state/current/x':
        'float',
    'traffic_light_state/current/y':
        'float',
    'traffic_light_state/current/z':
        'float',
    'traffic_light_state/past/state':
        'int',
    'traffic_light_state/past/valid':
        'int',
    'traffic_light_state/past/x':
        'float',
    'traffic_light_state/past/y':
        'float',
    'traffic_light_state/past/z':
        'float',
    'traffic_light_state/future/state':
        'int',
    'traffic_light_state/future/valid':
        'int',
    'traffic_light_state/future/x':
        'float',
    'traffic_light_state/future/y':
        'float',
    'traffic_light_state/future/z':
        'float',
}

features_description = {}
features_description.update(roadgraph_features)
features_description.update(state_features)
features_description.update(traffic_light_features)


# Example field definition
roadgraph_transforms = {
    'roadgraph_samples/dir':
        lambda x : np.reshape(x,(20000,3)),
    'roadgraph_samples/id':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/type':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/valid':
        lambda x : np.reshape(x,(20000,1)),
    'roadgraph_samples/xyz':
        lambda x : np.reshape(x,(20000,3)),
}

# Features of other agents.
state_transforms = {
    'state/id':
        lambda x : np.reshape(x,(128,)),
    'state/type':
        lambda x : np.reshape(x,(128,)),
    'state/is_sdc':
        lambda x : np.reshape(x,(128,)),
    'state/tracks_to_predict':
        lambda x : np.reshape(x,(128,)),
    'state/current/bbox_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/height':
        lambda x : np.reshape(x,(128,1)),
    'state/current/length':
        lambda x : np.reshape(x,(128,1)),
    'state/current/timestamp_micros':
        lambda x : np.reshape(x,(128,1)),
    'state/current/valid':
        lambda x : np.reshape(x,(128,1)),
    'state/current/vel_yaw':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/velocity_y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/width':
        lambda x : np.reshape(x,(128,1)),
    'state/current/x':
        lambda x : np.reshape(x,(128,1)),
    'state/current/y':
        lambda x : np.reshape(x,(128,1)),
    'state/current/z':
        lambda x : np.reshape(x,(128,1)),
    'state/future/bbox_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/height':
        lambda x : np.reshape(x,(128,80)),
    'state/future/length':
        lambda x : np.reshape(x,(128,80)),
    'state/future/timestamp_micros':
        lambda x : np.reshape(x,(128,80)),
    'state/future/valid':
        lambda x : np.reshape(x,(128,80)),
    'state/future/vel_yaw':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/velocity_y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/width':
        lambda x : np.reshape(x,(128,80)),
    'state/future/x':
        lambda x : np.reshape(x,(128,80)),
    'state/future/y':
        lambda x : np.reshape(x,(128,80)),
    'state/future/z':
        lambda x : np.reshape(x,(128,80)),
    'state/past/bbox_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/height':
        lambda x : np.reshape(x,(128,10)),
    'state/past/length':
        lambda x : np.reshape(x,(128,10)),
    'state/past/timestamp_micros':
        lambda x : np.reshape(x,(128,10)),
    'state/past/valid':
        lambda x : np.reshape(x,(128,10)),
    'state/past/vel_yaw':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/velocity_y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/width':
        lambda x : np.reshape(x,(128,10)),
    'state/past/x':
        lambda x : np.reshape(x,(128,10)),
    'state/past/y':
        lambda x : np.reshape(x,(128,10)),
    'state/past/z':
        lambda x : np.reshape(x,(128,10)),
}

traffic_light_transforms = {
    'traffic_light_state/current/state':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/valid':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/x':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/y':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/current/z':
        lambda x : np.reshape(x,(1,16)),
    'traffic_light_state/past/state':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/valid':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/x':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/y':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/past/z':
        lambda x : np.reshape(x,(10,16)),
    'traffic_light_state/future/state':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/valid':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/x':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/y':
        lambda x : np.reshape(x,(80,16)),
    'traffic_light_state/future/z':
        lambda x : np.reshape(x,(80,16)),
}

features_transforms = {}
features_transforms.update(roadgraph_transforms)
features_transforms.update(state_transforms)
features_transforms.update(traffic_light_transforms)

def transform_func(feature):
    transform = features_transforms
    keys = transform.keys()
    for key in keys:
        func = transform[key]
        feat = feature[key]
        feature[key] = func(feat)
    return feature

def WaymoDataset(dataroot):

    tfrecord_pattern = os.path.join(dataroot, 'tfrecords/{}')
    index_pattern = os.path.join(dataroot, 'idxs/{}')

    splits = {}
    fnlist = os.listdir(tfrecord_pattern.split('{}')[0])
    for fn in fnlist:
        splits[fn] = 1.

    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits, description=features_description, transform=transform_func)

    return dataset

def waymo_collate_fn(batch, GD, GS): # GS = max number of static roadgraph element (1400), GD = max number of dynamic roadgraph (16)
    past_states_batch = np.array([]).reshape(-1,10,2)
    past_states_mask_batch = np.array([]).reshape(-1,10)
    current_states_batch = np.array([]).reshape(-1,1,2)
    current_states_mask_batch = np.array([]).reshape(-1,1)
    future_states_batch = np.array([]).reshape(-1,80,2)
    future_states_mask_batch = np.array([]).reshape(-1,80)
    roadgraph_xyz_batch = np.array([]).reshape(-1,3)

    for data in batch:
        past_states = np.stack((data['state/past/x'],data['state/past/y']), axis=-1)
        past_states_mask = data['state/past/valid'] > 0.
        current_states = np.stack((data['state/current/x'],data['state/current/y']), axis=-1)
        current_states_mask = data['state/current/valid'] > 0.
        future_states = np.stack((data['state/future/x'],data['state/future/y']), axis=-1)
        future_states_mask = data['state/future/valid'] > 0.

        roadgraph_xyz = data['roadgraph_samples/xyz']

        past_states_batch = np.concatenate((past_states_batch, past_states), axis=0)
        past_states_mask_batch = np.concatenate((past_states_mask_batch, past_states_mask), axis=0)
        current_states_batch = np.concatenate((current_states_batch, current_states), axis=0)
        current_states_mask_batch = np.concatenate((current_states_mask_batch, current_states_mask), axis=0)
        future_states_batch = np.concatenate((future_states_batch, future_states), axis=0)
        future_states_mask_batch = np.concatenate((future_states_mask_batch, future_states_mask), axis=0)
        roadgraph_xyz_batch = np.concatenate((roadgraph_xyz_batch, roadgraph_xyz), axis=0)

        # future = np.stack(())

        
    return 0