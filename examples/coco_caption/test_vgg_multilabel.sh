#!/usr/bin/env bash

GPU_ID=1
WEIGHTS="/mnt/lvmhdd/xiaxiongwei/models/coco_img_captioning/tagging_vgg16_multilabel__iter_20000.caffemodel"
#WEIGHTS="/mnt/lvmhdd/xiaxiongwei/models/vgg/VGG_ILSVRC_16_layers.caffemodel"


./build/tools/caffe test \
    -model ./examples/coco_caption/tagging_vgg_multilabel.prototxt  \
    -weights $WEIGHTS \
    -gpu $GPU_ID
	#-snapshot /mnt/lvmhdd/xiaxiongwei/models/coco_img_captioning/tagging_vgg16_multilabel__iter_10000.solverstate \
