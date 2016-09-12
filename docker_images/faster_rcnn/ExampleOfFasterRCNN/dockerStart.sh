sudo nvidia-docker run -v models_container:/root/py-faster-rcnn/data/faster_rcnn_models \
	-v ${PWD}/examples:/root/examples \
	-v /storage/datasets/kitty/VersionForFasterRCNN/RadezVousWithKittiShort/:/root/py-faster-rcnn/data/RadezVousWithKittiShort/ \
        -v /storage/datasets/kitty/VersionForFasterRCNN/model/VGG_CNN_M_1024_FOUR_CLASSES/:/root/py-faster-rcnn/models/pascal_voc/VGG_CNN_M_1024_FOUR_CLASSES \
	-v /storage/datasets/kitty/imagenet_models:/root/py-faster-rcnn/data/imagenet_models \
	 --rm -i -t --net=host faster-rcnn-test /bin/bash
