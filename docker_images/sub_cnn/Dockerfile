# Copyright (c) 2016, deepsense.io
FROM nvidia/cuda:7.0-cudnn2-devel
MAINTAINER Rafal Hryciuk <rafal.hryciuk@deepsense.io>
# Install Oracle Java 8
RUN apt-get -y update && apt-get -y install software-properties-common
RUN \
  echo oracle-java8-installer shared/accepted-oracle-license-v1-1 select true | debconf-set-selections && \
  add-apt-repository -y ppa:webupd8team/java && \
  apt-get update && \
  apt-get install -y oracle-java8-installer && \
  rm -rf /var/lib/apt/lists/* && \
  rm -rf /var/cache/oracle-jdk8-installer
# Install numpy
RUN apt-get update && apt-get install -y python-numpy python-scipy python-matplotlib ipython ipython-notebook python-pandas python-sympy python-nose
# Install Linux packages and tools for ML
RUN apt-get -y update && apt-get install -y \
    gfortran \
    libopenblas-dev \
    liblapack-dev
# Install Python
RUN apt-get update && apt-get install -y python python-pip python-dev
RUN apt-get update && apt-get install -y libfreetype6-dev libxft-dev
# Install Python packages for ML
RUN pip install -U setuptools
RUN pip install \
    bokeh==0.10 \
    bunch \
    enum34 \
    MarkupSafe \
    matplotlib \
    mock \
    pandas \
    pymongo \
    setproctitle \
    sklearn \
    web.py \
    tabulate \
    termcolor \
    Theano \
    xxhash
RUN pip install --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow-0.9.0rc0-cp27-none-linux_x86_64.whl
RUN pip install pandas --upgrade
RUN apt-get -y update && apt-get install -y python-skimage
# Install other necessary packages
RUN apt-get -y update && apt-get install -y python-opencv
# Upgrade necessary packages
RUN pip install setuptools==20.3.1
RUN apt-get -y update && apt-get -y build-dep python-imaging
RUN apt-get -y update && apt-get install -y libjpeg8 libjpeg62-dev libfreetype6 libfreetype6-dev
RUN pip install -U scikit-image
RUN sudo easy_install --upgrade numpy
RUN sudo easy_install --upgrade scipy
RUN pip install keras
# Maciek's additions
RUN apt-get -y install python-opencv
RUN apt-get -y install libopencv-dev
# RUN apt-get -y install boost # Doesn't work
RUN apt-get -y install libboost-dev
# RUN apt-get -y install libgflags-de # Typo?
RUN apt-get -y install libgflags-dev
RUN apt-get -y install libgoogle-glog-dev
RUN apt-get -y install libprotobuf-dev
RUN apt-get -y install libhdf5-dev
RUN apt-get -y install libleveldb-dev
RUN apt-get -y install liblmdb0
RUN apt-get -y install liblmdb-dev
RUN apt-get -y install libsnappy1
RUN apt-get -y install libsnappy-dev
RUN apt-get -y install libboost-all-dev
RUN apt-get -y install vim
RUN pip install EasyDict
# RUN pip install python-opencv # No such thing
RUN apt-get -y install python-opencv
RUN pip install cython
RUN ln -s /usr/local/nvidia/lib64/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so
# Caffe
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        libatlas-base-dev \
        libgflags-dev \
        libhdf5-serial-dev \
        liblmdb-dev \
        protobuf-compiler \
	mc bc htop
# && \
#rm -rf /var/lib/apt/lists/*
# RUN hostname subcnn
RUN mkdir /code
WORKDIR /code
RUN git clone --recursive https://github.com/yuxng/SubCNN.git && \
    export ROOT=/code/SubCNN && \
    cd $ROOT/fast-rcnn/lib && \
    cat setup.py | sed 's/sm_35/sm_30/' > setup.py.tmp && \
    mv setup.py.tmp setup.py && \
    make && \
    cd $ROOT/caffe-fast-rcnn && \
    cat Makefile.config.example | sed "s/^# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/" | sed "s/^# USE_CUDNN := 1/USE_CUDNN := 1/" > Makefile.config && \
    echo "CUSTOM_CXX := g++ -std=c++11" >> Makefile.config && \ 
    make -j40 && make pycaffe
VOLUME /kitti
RUN ln -s /kitti/imagenet_models $ROOT/fast-rcnn/data/imagenet_models 
RUN ln -s /kitti/data_object_image_2 $ROOT/fast-rcnn/data/KITTI/data_object_image_2
RUN ln -s /kitti/voxel_examplars $ROOT/fast-rcnn/data/KITTI/voxel_examplars
#ENV CLONE_TAG=master
#RUN git clone -b ${CLONE_TAG} --depth 1 https://github.com/BVLC/caffe.git . && \
#    for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
#    mkdir build && cd build && \
#    cmake ${CMAKE_ARGS} .. && \
#    make -j"$(nproc)"
#ENV PYCAFFE_ROOT $CAFFE_ROOT/python
#ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
#ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
#RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig
## WORKDIR /workspace # Caffe used it
#RUN cat Makefile.config.example | sed "s/^# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/" | sed "s/^# USE_CUDNN := 1/USE_CUDNN := 1/" > Makefile.config
#RUN echo "ln -s /dev/null /dev/raw1394" >> /root/.bashrc
#RUN make all -j12
#RUN make test -j12
## you can use this command manually to test caffe:
## /bin/bash -c 'source /root/.bashrc; make runtest'
## Faster-RCNN
#RUN git clone --recursive https://github.com/rbgirshick/py-faster-rcnn.git /root/py-faster-rcnn/
#WORKDIR /root/py-faster-rcnn
#RUN sed 's/sm_35/sm_30/' lib/setup.py > lib/setup.py.tmp
#RUN mv lib/setup.py.tmp lib/setup.py
#RUN sed 's/__C.USE_GPU_NMS = True/__C.USE_GPU_NMS = False/' lib/fast_rcnn/config.py > lib/fast_rcnn/config.py.tmp
#RUN mv lib/fast_rcnn/config.py.tmp lib/fast_rcnn/config.py
#RUN cat caffe-fast-rcnn/Makefile.config.example | sed "s/^# WITH_PYTHON_LAYER := 1/WITH_PYTHON_LAYER := 1/" | sed "s/^# USE_CUDNN := 1/USE_CUDNN := 1/" > caffe-fast-rcnn/Makefile.config
#RUN make -C /root/py-faster-rcnn/lib/
#RUN make -j8 -C /root/py-faster-rcnn/caffe-fast-rcnn/
#RUN make pycaffe -C /root/py-faster-rcnn/caffe-fast-rcnn/
## you can use these commands manually to test faster-rcnn:
## ./data/scripts/fetch_faster_rcnn_models.sh
## ./tools/demo.py # with commented lines calling vis_detections (L98) and plt.show() (L151)

