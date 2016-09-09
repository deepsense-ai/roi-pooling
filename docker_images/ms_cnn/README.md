## Running a demo docker with Caffe and Faster R-CNN installed

0. Log in as root to ml1, ml2 or ml3

1. If your docker already exists switch to step 6

2. In the directory where the Dockerfile is located execute `sudo docker build .` 

3. run  the command `sudo docker images` and check the id of your image

4. tag your docker by running the following command: `sudo docker tag [your_id] [your_tag] `

5. run the docker
`nvidia-docker run -v /mnt/storage/datasets/kitty/data_object_image_2/training:/kitti -i -t --net=host [your_tag] /bin/bash`

6. inside the docker: 
..* run `cd mscnn/examples/kitti_car/mscnn-7s-576-2x`

..* in the files trainval_1st.prototxt and trainval_2nd.prototxt replace the lines "../../../data/kitti/window_files/mscnn_window_file_kitti_vehicle_train.txt"
 and "../../../data/kitti/window_files/mscnn_window_file_kitti_vehicle_val.txt" with "/kitti/window_files/mscnn_window_file_kitti_vehicle_train.txt" and
"/kitti/window_files/mscnn_window_file_kitti_vehicle_val.txt"

..* in the file train_mscnn.sh replace ../../../models/VGG/VGG_ILSVRC_16_layers.caffemodel with /kitti/VGG/VGG_ILSVRC_16_layers.caffemodel, optionally you can change gpu id

..* run the training by executing `sh train_mscnn.sh`

