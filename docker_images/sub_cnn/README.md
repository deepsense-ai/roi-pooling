## Running a demo docker with Caffe and Faster R-CNN installed

Not ideal, but does the job - it seems that three symbolic links still do not work. Just in the meantime add them manually. 

0. check that image `localhost/caffe` exists - it should on `ml1`.
If it doesn't exist you need to build it.

1. run the docker
`sudo nvidia-docker run -v /mnt/storage/datasets/kitty:/kitti -i -t --net=host localhost/subcnn_gamma /bin/bash`

2. leave the console with ctrl-p + ctrl-q

3. return to the console with 

```
docker ps (this is just to  check the instance name)
docker attach [instance_name]
```

4. ...or with 

```
docker exec -i -t [instance_name] /bin/bash
```
