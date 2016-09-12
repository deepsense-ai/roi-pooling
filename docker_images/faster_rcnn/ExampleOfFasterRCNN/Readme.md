##Running an example of Faster-RCNN

1. Build a docker instance using the instructions in the superfolder.
2. Edit dockerStart.sh replacing faster-rcnn-test with the name of your docker.
3. Run ./dockerStart.sh
4. In docker cd ~/examples
5. Run trainShortKitti.sh and later demoShortKitti.sh

Further information:

1. trainLongKitti.sh Invokes train_faster_rcnn_alt_opt.py with the following parameters:
  --gpu 0 \ (obvious)
  --net_name "VGG_CNN_M_1024_FOUR_CLASSES" \ (the folder with parameters and architectures of networks to be train.
                            It is a modification of VGG_CNN_M_1024, which support 4 classes (instead standard 20).
  --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \ (initialization from the imagenet)
  --imdb RadezVousWithKittiShort \ (the dataset to be described)
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \ (some parameters)
  --iterations "100 100 100 100" (number of iterations in the four stages training described in the paper)

2. kittiReader.py is the dataloder, cloned from the pascal voc dataloard, with changed the list of classes to "car, track, van, tram" and jpg to png in the format specification. The loader is registered in factory.py

3. demoShortKitti.sh invokes demoMy.py with the model trained above. It scans the folder given in --dir and for each detection produces a file of the name {detected class name}{original name}.

The dataset RadezVousWithKittiShort is located in /storage/datasets/kitty/VersionForFasterRCNN/RadezVousWithKittiShort and contains only a few files. For any picture in PNGImages there is an xml file in Annotations. The section <object></object> contains the description of the RoIs on a given picture. The file ImageSets/Main/trainval.txt contains the list of images used for training.
