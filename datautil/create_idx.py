from subprocess import call
import os.path
import glob

tfrecord_path = './data/tfrecords'
idx_path = './data/idxs'
batch_size = 1

for tfrecord in glob.glob(tfrecord_path+'/*'):
    idxname = idx_path + '/' + tfrecord.split('/')[-1]
    call(["tfrecord2idx", tfrecord, idxname])
