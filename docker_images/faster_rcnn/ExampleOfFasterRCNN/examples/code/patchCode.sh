#!/bin/bash


DEST_DIR="/root/py-faster-rcnn/"
SRC_DIR="/root/examples/code/"

cp "${SRC_DIR}kittiReader.py" "${DEST_DIR}lib/datasets/"
cp "${SRC_DIR}factory.py" "${DEST_DIR}lib/datasets/"
cp "${SRC_DIR}trainShortKitti.sh" "${DEST_DIR}"
cp "${SRC_DIR}trainLongKitti.sh" "${DEST_DIR}"
cp "${SRC_DIR}demoShortKitti.sh" "${DEST_DIR}"
cp "${SRC_DIR}train_faster_rcnn_alt_opt.py" "${DEST_DIR}/tools/"
cp "${SRC_DIR}demoMy.py" "${DEST_DIR}/tools/"
