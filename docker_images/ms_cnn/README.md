## Running a demo docker with Caffe and Faster R-CNN installed

0. check that image `localhost/caffe` exists - it should on `ml1`.
If it doesn't exist you need to build it.

1. run the docker
`sudo nvidia-docker run -v models_container:/root/py-faster-rcnn/data/faster_rcnn_models --rm -i -t --net=host localhost/caffe /bin/bash`

2. inside the docker execute 

```
./data/scripts/fetch_faster_rcnn_models.sh  # no need to re-download, models will be stored in a volume
./tools/demo.py # with commented lines calling vis_detections (L98) and plt.show() (L151)
```
