from subprocess import call
import os.path
import glob
from tqdm import tqdm

tfrecord_path = './data/scenario/validation'
idx_path = './data/idxs_validation_bs_2'
batch_size = 1

for tfrecord in tqdm(glob.glob(tfrecord_path+'/*')):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    call(["tfrecord2idx", tfrecord, idxname])
