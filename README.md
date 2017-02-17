# roi-pooling

This repo contains the implementation of Region of Interest Pooling as a custom TensorFlow operation. The CUDA code responsible for the computations was largely taken from the original Caffe implementation by Ross Girshick.

# Requirements

You need to have CUDA and TensorFlow installed on your system to compile and use the operation.

# Build

To compile the operation, issue the following commands:

```
cd src
make
```

# Usage

For usage example please see the example directory.
