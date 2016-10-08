# -------------------------------------------------------------------
# Create the LMDB for the data instances
# Both train and validation lmdbs can be created using this 
# The file is adapted from BVLC Caffe, and requires Caffe tools
# Author: Sukrit Shankar 
# -------------------------------------------------------------------
set='val'
data_list_name='dataset_list_val.txt'
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
# Please set the appropriate paths
EXAMPLE=/mnt/lvm/xiaxiongwei/multi-label_dataset/data_lmdb  			# Path where the output LMDB is stored
#EXAMPLE=../data/coco/data_lmdb  			# Path where the output LMDB is stored
DATA=../../dataset       			# Path where the data.txt file is present 
TOOLS=../build/tools    			# Caffe dependency to access the convert_imageset utility 
DATA_ROOT=/mnt/lvmhdd/xiaxiongwei/workspace/tagging/images/			# Path prefix for each entry in data.txt
# +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

# Creating LMDB
 echo "Creating data lmdb..."
 GLOG_logtostderr=1 $TOOLS/convert_imageset \
 	--resize_height=256 \
	--resize_width=256 \
    $DATA_ROOT \
    $DATA/$data_list_name \
    $EXAMPLE/$set

# ------------------------------
echo "Done."



