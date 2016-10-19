import unittest
import tempfile
import os
import numpy as np

import caffe
GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.
caffe.set_mode_gpu()
caffe.set_device(GPU_ID)


import tensorflow as tf
import numpy as np
from roi_pooling_ops import roi_pooling


def roinet_file(pooled_w, pooled_h, n_rois):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""
        name: 'roi pooling net' force_backward: true
        input: 'data' input_shape {{ dim: 1 dim: 1 dim: 4 dim: 5 }}
        
        input: 'rois'
        input_shape {{
          dim: {n_rois} # to be changed on-the-fly to num ROIs
          dim: 5 # [batch ind, x1, y1, x2, y2] zero-based indexing
        }}

        layer {{
          name: "roi_pool5"
          type: "ROIPooling"
          bottom: "data"
          bottom: "rois"
          top: "output"
          roi_pooling_param {{
            pooled_w: {pooled_w}
            pooled_h: {pooled_h}
            #spatial_scale: 0.0625 # 1/16
            spatial_scale: 1 # 1/16
          }}
          }}""".format(pooled_w=pooled_w, pooled_h=pooled_h, n_rois=n_rois))
        return f.name
    
def ROI_caffe(x, rois, pooled_w, pooled_h):
    #TODO: maybe check the initial conditions for the ROIs, input etc.?
    
    net_file = roinet_file(pooled_w=pooled_w, pooled_h=pooled_h, n_rois=len(rois))
    net = caffe.Net(net_file, caffe.TRAIN)
    os.remove(net_file)
    
    net.blobs['data'].data[...] = x
    net.blobs['rois'].data[...]  = rois
    net.forward()
    output = net.blobs['output'].data
    
    net.blobs['output'].diff[...] = 1
    net.backward()
    caffe_grad = net.blobs['data'].diff
    del net
    return output, caffe_grad

def ROI_tensorflow(x_input, rois_input, pooled_w, pooled_h):
    input = tf.placeholder(tf.float32)
    rois = tf.placeholder(tf.int32)
    
    y = roi_pooling(input, rois, pool_height=2, pool_width=2)
    #mean = tf.reduce_mean(y)
    mean = tf.reduce_sum(y)
    grads = tf.gradients(mean, input)
    with tf.Session('') as sess:
        with tf.device('/gpu:3'):
            y_output =  sess.run(y, feed_dict={input: x_input, rois: rois_input})
            grads_output = sess.run(grads, feed_dict={input: x_input, rois: rois_input})
    return y_output, grads_output

x = np.arange(20).reshape(1, 1, 4, 5)
rois = [[0, 0, 0, 1, 1],
        [0, 0, 0, 3, 3],
        [0, 2, 2, 3, 3],
        [0, 0, 0, 4, 3]]
rois = np.array(rois)

if False:
    caffe_y, caffe_grad = ROI_caffe(x, rois, 2, 2)
    tf_y, tf_grad = ROI_tensorflow(x, rois, 2, 2)
    
    print('======== ROI outputs: ==========')
    print('tf output: ', tf_y)
    print('caffe output: ', caffe_y)
    
    printprint('======== ROI gradients: ==========')
    print('tf output: ', tf_grad)
    print('caffe output: ', caffe_grad)
