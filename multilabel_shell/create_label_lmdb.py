# -------------------------------------------------------------------
# Create the LMDB for the labels
# Both train and validation lmdbs can be created using this 
# Author: Sukrit Shankar 
# -------------------------------------------------------------------

# -------------------------------------
import pylab as pltss
from pylab import *
import numpy as np
import matplotlib.pyplot as plt 
import scipy 
import scipy.io
import os.path
import lmdb	 

# -------- Import Caffe ---------------
caffe_root = '../' 
import sys 
sys.path.insert(0, caffe_root + 'python')
import caffe




labels_mat_file = '../../test_final.mat'							# Mat file for labels N x M 
mat_contents = scipy.io.loadmat(labels_mat_file)
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Please set the following values and paths as per your needs 
N = mat_contents['label'].shape[0] 
M = 235									# Number of possible labels for each data instance 
output_lmdb_path = '../data/coco/label_lmdb/test/'   	# Path of the output label LMDB
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+


# -------- Write in LMDB for Caffe ----------
X = np.zeros((N, M, 1, 1), dtype=np.uint8)
y = np.zeros(N, dtype=np.int64)
map_size = X.nbytes * 10
env = lmdb.open(output_lmdb_path, map_size=map_size)

# ---------------------------------
# Read the mat file and assign to X
X[:,:,0,0] = mat_contents['label'] 	
# The above expects that the MAT file contains the variable as labels	
# To instead check the variable names in the mat file, and use them in a more judicious way, do 
# array_names = scipy.io.whosmat(labels_mat_file) 	
# print '\n Array Names \n', array_names
print X						# Check to see if the contents are well populated within the expected range
print X.shape					# Check to see if X is of shape N x M x 1 x 1     

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        if i%100000==0:
            print i
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = X.shape[1]
        datum.height = X.shape[2]
        datum.width = X.shape[3]
        datum.data = X[i].tostring()  	# or .tobytes() if numpy < 1.9 
        datum.label = int(y[i])
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())

	# Print the progress 
	print 'Done Label Writing for Data Instance = ' + str(i)




