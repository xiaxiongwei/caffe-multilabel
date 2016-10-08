#!/usr/bin/env bash

GPU_ID=0
WEIGHTS="/mnt/lvmhdd/xiaxiongwei/models/googlenet/imagenet_gnet_iter_78816.caffemodel"


./build/tools/caffe train \
    -solver ./examples/coco_caption/tag_solver.gnet.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
	#-snapshot /mnt/lvmhdd/xiaxiongwei/models/coco_img_captioning/tagging_gnet_multilabel_235_adam_iter_60000.solverstate \
