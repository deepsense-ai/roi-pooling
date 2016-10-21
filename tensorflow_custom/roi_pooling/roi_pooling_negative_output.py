import tensorflow as tf
from roi_pooling_ops import roi_pooling
import os
import numpy as np

import sys
n_rois = int(sys.argv[1])


def compute_ROI(x_input, rois_input, pooled_w, pooled_h):
    input = tf.placeholder(tf.float32)
    rois = tf.placeholder(tf.int32)
    
    y  = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)
    with tf.Session('') as sess:
        with tf.device('/gpu:0'):
            y_output =  sess.run(y, feed_dict={input: x_input, rois: rois_input})
    return y_output, None

data_dir = '/storage/kdziedzic/roi_pool_input/'
input_path = os.path.join(data_dir, 'input_layer.npy')
rois_path = os.path.join(data_dir, 'input_rois.npy')

rois = np.load(rois_path)
rois_sampled = rois[:n_rois]

input = np.load(input_path)

print 'input negatives: ', (input < 0).mean()
print 'number of rois: ', n_rois

y, grad = compute_ROI(input, rois_sampled, 7, 7)
print 'y negatives: ', (y < 0 ).mean()
print 'y shape: ', y.shape
