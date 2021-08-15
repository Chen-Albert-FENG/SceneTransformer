from subprocess import call
import os.path
import glob
from tqdm import tqdm

tfrecord_path = './data/tf_example/training'
idx_path = './data/idxs_training_bs_4'
batch_size = 4

for tfrecord in tqdm(glob.glob(tfrecord_path+'/*')):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    call(["tfrecord2idx", tfrecord, idxname])
