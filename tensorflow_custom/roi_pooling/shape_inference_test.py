import unittest
import tensorflow as tf
import numpy as np
from roi_pooling_ops import roi_pooling

class RoiPoolingShapeInferenceTest(unittest.TestCase):
    
    def test_basic_shape_inference(self):
        pooled_w, pooled_h = 2, 2
        input_w, input_h = 200, 200
        n_channels = 3
        n_batches = None
        input = tf.placeholder(tf.float32, shape=[n_batches, n_channels, input_w, input_h])
        
        n_rois = None
        single_roi_dimension = 5
        rois = tf.placeholder(tf.int32, shape=[n_rois, single_roi_dimension])
        
        y = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)
        
        print 'output shape: ', y.get_shape()
        
        self.assertEqual(y.get_shape().ndims, 4)
        self.assertIs(y.get_shape()[0].value, n_rois)
        self.assertIs(y.get_shape()[1].value, n_channels)
        self.assertIs(y.get_shape()[2].value, pooled_w)
        self.assertIs(y.get_shape()[3].value, pooled_h)

    
#    def test_reference(self):
#        M = 2
#        outputs = 1
#        a = tf.placeholder(tf.float32, shape=(None, M))
#        b = tf.placeholder(tf.float32, shape=(M, outputs))
#        y = tf.matmul(a,b)
#        self.assertIs(y.get_shape()[0].value, None)
#        self.assertEqual(y.get_shape()[1].value, outputs)
        
if __name__ == '__main__':
    unittest.main()
    
    