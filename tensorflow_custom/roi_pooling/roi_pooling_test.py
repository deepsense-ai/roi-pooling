import tensorflow as tf
import numpy as np
from roi_pooling_ops import roi_pooling

class RoiPoolingTest(tf.test.TestCase):
    # TODO(maciek): add python, implementation and test outputs
    # TODO(maciek): test pool_height != pool_width, height != width

    def testRoiPoolingGrad_1(self):
        # TODO(maciek): corner cases
        input_value = [[[[1, 2, 4, 4],
                 [3, 4, 1, 2],
                 [6, 2, 1, 7.0],
                 [1, 3, 2, 8]]]]
        input_value = np.asarray(input_value, dtype='float32')

        rois_value = [
                      [0, 0, 0, 1, 1],
                      [0, 1, 1, 2, 2],
                      [0, 2, 2, 3, 3],
                      [0, 0, 0, 2, 2],
                      [0, 0, 0, 3, 3]
            ]
        rois_value = np.asarray(rois_value, dtype='int32')
        with tf.Session(''):
            # NOTE(maciek): looks like we have to use consts here, based on tensorflow/python/ops/nn_test.py
            input_const = tf.constant(input_value, tf.float32)
            rois_const = tf.constant(rois_value, tf.int32)
            y = roi_pooling(input_const, rois_const, pool_height=2, pool_width=2)
            mean = tf.reduce_mean(y)

            numerical_grad_error_1 = tf.test.compute_gradient_error([input_const], [input_value.shape],
                                                      y, [5, 1, 2, 2])

            numerical_grad_error_2 = tf.test.compute_gradient_error([input_const], [input_value.shape],
                                                      mean, [])
            print numerical_grad_error_1, numerical_grad_error_2
            self.assertLess(numerical_grad_error_1, 1e-4)
            self.assertLess(numerical_grad_error_2, 1e-4)

if __name__ == '__main__':
    tf.test.main()
