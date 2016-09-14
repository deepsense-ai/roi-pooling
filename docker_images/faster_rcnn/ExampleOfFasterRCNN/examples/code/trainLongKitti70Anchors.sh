#!/bin/bash

LOG="experiments/logs/faster_rcnn_alt_opt_${NET}_${EXTRA_ARGS_SLUG}.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"

time ./tools/train_faster_rcnn_alt_opt.py --gpu 2 \
  --net_name "VGG_CNN_M_1024_FOUR_CLASSES" \
  --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \
  --imdb RandezVousWithKitti \
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \
  --iterations "10000 10000 10000 10000" \
  --70Anchors "True"
set +x
NET_FINAL=`grep "Final model:" ${LOG} | awk '{print $3}'`
set -x

