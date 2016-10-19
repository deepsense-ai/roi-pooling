import unittest
import tempfile
import os
import numpy as np

import tensorflow as tf
import numpy as np
from roi_pooling_ops import roi_pooling

from generate_caffe_reference_data import load_data

def ROI_tensorflow(x_input, rois_input, pooled_w, pooled_h):
    input = tf.placeholder(tf.float32)
    rois = tf.placeholder(tf.int32)
    
    y = roi_pooling(input, rois, pool_height=2, pool_width=2)
    mean = tf.reduce_sum(y)
    grads = tf.gradients(mean, input)
    with tf.Session('') as sess:
        with tf.device('/gpu:3'):
            y_output =  sess.run(y, feed_dict={input: x_input, rois: rois_input})
            grads_output = sess.run(grads, feed_dict={input: x_input, rois: rois_input})
            grads_output = grads_output[0]
    return y_output, grads_output


class RoiReferenceCaffeTest(unittest.TestCase): 
    def test_equal_to_caffe(self):
        (x, rois, caffe_y, caffe_grad) = load_data()
        tf_y, tf_grad = ROI_tensorflow(x, rois, 2, 2)
            
        print '======== ROI outputs: =========='
        print 'tf output: \n', tf_y
        print 'caffe output: \n', caffe_y
        	  
        print '======== ROI gradients: =========='
        print 'tf grad: \n', tf_grad
        print 'caffe grad: \n', caffe_grad
        
        np.testing.assert_almost_equal(actual=tf_y, desired=caffe_y, decimal=10)
        np.testing.assert_almost_equal(actual=tf_grad, desired=caffe_grad, decimal=10)

if __name__ == '__main__':
    unittest.main()