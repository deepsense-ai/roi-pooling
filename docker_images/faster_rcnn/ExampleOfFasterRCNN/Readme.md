#Training ''Kitti'' with  Faster-RCNN

1. Build a docker instance using the instructions in the superfolder.
2. Edit dockerStart.sh replacing faster-rcnn-test with the name of your docker.
3. Run ./dockerStart.sh
4. In docker cd ~/examples
5. Run trainShortKitti.sh and later demoShortKitti.sh to get a short test.
6. Run trainLongKitti.sh and demoLongKitti.sh for a full training (note that trainLongKitti.sh my seem unresponsive at the begining as it process the dataset before the acctual training. How about having a cup of coffe? ;))

Further information:

1. trainShortKitti.sh Invokes train_faster_rcnn_alt_opt.py with the following parameters:
  --gpu 0 \ (obvious)
  --net_name "VGG_CNN_M_1024_FOUR_CLASSES" \ (the folder with parameters and architectures of networks to be train.
                            It is a modification of VGG_CNN_M_1024, which support 4 classes (instead standard 20).
  --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel \ (initialization from the imagenet)
  --imdb RadezVousWithKittiShort \ (the dataset to be described)
  --cfg experiments/cfgs/faster_rcnn_alt_opt.yml \ (some parameters)
  --iterations "100 100 100 100" (number of iterations in the four stages training described in the paper)

2. kittiReader.py is the dataloder, cloned from the pascal voc dataloader, with changed the list of classes to "car, track, van, tram" and jpg to png in the format specification. 

3. Datasets need to be registered in factory.py 

4. demoShortKitti.sh invokes demoMy.py with the model trained above. It scans the folder given in --dir and for each detection produces a file of the name {detected class name}{original name}.

Datasets:
1. RandezVousWithKittiShort is located in /storage/datasets/kitty/VersionForFasterRCNN/RandezVousWithKittiShort and contains only a few files. For any picture in PNGImages there is an xml file in Annotations. The section <object></object> contains the description of the RoIs on a given picture. The file ImageSets/Main/trainval.txt contains the list of images used for training.
2. RandezVousWithKitti - full kitti dataset of 7481 images located in /storage/datasets/kitty/VersionForFasterRCNN/RandezVousWithKitti. The file trainval.txt contains the list used for training, consisting of 5985 images (use trainLongKitti.sh to run the training process). The file validation.txt contains the list of 1496 files to be used in validation (currently it is only for information and is not used by code). The file tranvalAll.txt contains the list of all 7481 files (it is not used by the code). The directory Validation contains files listed in validation.txt (used by demoLongKitti.sh). 

Remaks:
1.
2. Logs are stored in py-faster-rcnn/experiments/logs/
3. The code does not check for a free gpu. If you get Check failed: error == cudaSuccess (2 vs. 0)  out of memory

root/py-faster-rcnn/output/faster_rcnn_alt_opt/kitti_trainval

py-faster-rcnn/data/cache

