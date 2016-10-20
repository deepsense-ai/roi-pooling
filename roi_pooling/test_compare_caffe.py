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
    
    y = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)
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
        
        self.assertEqual(x.ndim, 4)
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(x.shape[1], 1)
        
        self.assertEqual(rois.ndim, 2)
        self.assertEqual(rois.shape[1], 5)
        
        p_width = 7
        p_height = 7
        tf_y, tf_grad = ROI_tensorflow(x, rois, p_width, p_height)

        self.assertEqual(tf_grad.ndim, 4)
        self.assertEqual(tf_grad.shape[0], 1)
        self.assertEqual(tf_grad.shape[1], 1)
        
        self.assertEqual(tf_grad.shape[2], x.shape[2])
        self.assertEqual(tf_grad.shape[3], x.shape[3])
        
        self.assertEqual(tf_y.ndim, 4)
        self.assertEqual(tf_y.shape[1], 1)
        self.assertEqual(tf_y.shape[2], p_width)
        self.assertEqual(tf_y.shape[3], p_height)
            
        self.assertEqual(tf_y.shape[0], rois.shape[0])
        self.assertEqual(tf_grad.sum(), caffe_grad.sum())
                
        print '======== ROI outputs: =========='
        print 'tf output: \n', tf_y
        print 'caffe output: \n', caffe_y
        	  
        print '======== ROI gradients: =========='
        print 'tf grad: \n', tf_grad
        print 'caffe grad: \n', caffe_grad
        
        print 'tf grad sum: ', tf_grad.sum()
        print 'caffe grad sum: ', caffe_grad.sum()
        
        
        np.testing.assert_almost_equal(actual=tf_y, desired=caffe_y, decimal=10)
        np.testing.assert_almost_equal(actual=tf_grad, desired=caffe_grad, decimal=10)

if __name__ == '__main__':
    unittest.main()