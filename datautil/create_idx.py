from subprocess import call
import os.path
import glob

# test_data_root = os.environ['DALI_EXTRA_PATH']
tfrecord_path = '/home/user/Projects/scene_transformer/data/tfrecords'
idx_path = '/home/user/Projects/scene_transformer/data/idxs'
batch_size = 16

for tfrecord in glob.glob(tfrecord_path+'/*'):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    call(["tfrecord2idx", tfrecord, idxname])
