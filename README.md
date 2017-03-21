# roi-pooling

<div style="text-align:center"><img src="https://deepsense.io/wp-content/uploads/2017/02/roi_pooling-1.gif" width="600"/></div>

This repo contains the implementation of Region of Interest Pooling as a custom TensorFlow operation. The CUDA code responsible for the computations was largely taken from the original Caffe implementation by Ross Girshick.

For more information about RoI pooling you can check out our [blog](https://deepsense.io/region-of-interest-pooling-explained/)

# Requirements

You need to have CUDA and TensorFlow  installed on your system to compile and use the operation. The code was tested with CUDA 8.0 and TensorFlow 0.12.0 and 1.0.0.

# Install

To compile and install the operation, issue the following commands:

```
python setup.py install
```
