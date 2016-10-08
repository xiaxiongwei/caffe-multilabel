#!/usr/bin/env bash

GPU_ID=3
WEIGHTS="/mnt/lvmhdd/xiaxiongwei/models/vgg/VGG_ILSVRC_16_layers.caffemodel"


./build/tools/caffe train \
    -solver ./examples/coco_caption/tag_solver.vgg.prototxt \
    -weights $WEIGHTS \
    -gpu $GPU_ID
	#	-snapshot /mnt/lvmhdd/xiaxiongwei/models/coco_img_captioning/tagging_vgg16_multilabel_235_uniform_adam_iter_31072.solverstate \
