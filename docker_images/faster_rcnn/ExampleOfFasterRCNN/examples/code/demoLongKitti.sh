#!/bin/bash

./tools/demoMy.py --gpu 0 --net_protocol VGG_CNN_M_1024_FOUR_CLASSES \
	--net_model /root/py-faster-rcnn/output/faster_rcnn_alt_opt/kitti_trainval/VGG_CNN_M_1024_FOUR_CLASSES_faster_rcnn_final.caffemodel \
	--dir /root/py-faster-rcnn/data/RandezVousWithKitti/Data/Validation
