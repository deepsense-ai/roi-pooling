#!/bin/bash


echo "Patching code"

/root/examples/code/patchCode.sh

cd /root/py-faster-rcnn

echo "Running train example"
./demoShortKitti.sh
