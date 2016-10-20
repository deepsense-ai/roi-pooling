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

    def test_with_reference_output(self):
        def load_reference_data():
            input_image_file = 'test_input.npy'
            input_roi_file = 'test_roi.npy'
            caffe_output_file = 'test_caffe_output.npy'
            caffe_grad_file =  'test_caffe_grad.npy'
            
            filenames = [input_image_file, input_roi_file, caffe_output_file, caffe_grad_file]
            return [np.load(f) for f in filenames]
        
        def compute_ROI(x_input, rois_input, pooled_w, pooled_h):
            input = tf.placeholder(tf.float32)
            rois = tf.placeholder(tf.int32)
            
            y = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)
            mean = tf.reduce_sum(y)
            grads = tf.gradients(mean, input)
            with tf.Session('') as sess:
                y_output =  sess.run(y, feed_dict={input: x_input, rois: rois_input})
                grads_output = sess.run(grads, feed_dict={input: x_input, rois: rois_input})
                grads_output = grads_output[0]
            return y_output, grads_output

        (x, rois, reference_y, reference_grad) = load_reference_data()
        
        self.assertEqual(x.ndim, 4)
        self.assertEqual(x.shape[0], 1)
        self.assertEqual(x.shape[1], 1)
        
        self.assertEqual(rois.ndim, 2)
        self.assertEqual(rois.shape[1], 5)
        
        p_width = 7
        p_height = 7
        tf_y, tf_grad = compute_ROI(x, rois, p_width, p_height)

        self.assertEqual(tf_grad.ndim, 4)
        self.assertEqual(tf_grad.shape[0], 1)
        self.assertEqual(tf_grad.shape[1], 1)
        
        self.assertEqual(tf_grad.shape[2], x.shape[2])
        self.assertEqual(tf_grad.shape[3], x.shape[3])
        
        self.assertEqual(tf_y.ndim, 4)
        self.assertEqual(tf_y.shape[1], 1)
        self.assertEqual(tf_y.shape[2], p_height)
        self.assertEqual(tf_y.shape[3], p_width)
            
        self.assertEqual(tf_y.shape[0], rois.shape[0])
        self.assertEqual(tf_grad.sum(), reference_grad.sum())
                
        np.testing.assert_almost_equal(actual=tf_y, desired=reference_y,
                                       decimal=10)
        np.testing.assert_almost_equal(actual=tf_grad, desired=reference_grad,
                                       decimal=10)

    def test_shape_inference1(self):
        pooled_w, pooled_h = 2, 2
        input_w, input_h = 200, 200
        n_channels = 3
        n_batches = None
        input = tf.placeholder(tf.float32, shape=[n_batches, n_channels, input_w, input_h])
        
        n_rois = None
        single_roi_dimension = 5
        rois = tf.placeholder(tf.int32, shape=[n_rois, single_roi_dimension])
        
        y = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)
        
        self.assertEqual(y.get_shape().ndims, 4)
        self.assertIs(y.get_shape()[0].value, n_rois)
        self.assertIs(y.get_shape()[1].value, n_channels)
        self.assertIs(y.get_shape()[2].value, pooled_h)
        self.assertIs(y.get_shape()[3].value, pooled_w)

    def test_shape_inference2(self):
        pooled_w, pooled_h = 3, 4
        input_w, input_h = 200, 300
        n_channels = 3
        n_batches = None
        input = tf.placeholder(tf.float32, shape=[n_batches, n_channels, input_w, input_h])
        
        n_rois = None
        single_roi_dimension = 5
        rois = tf.placeholder(tf.int32, shape=[n_rois, single_roi_dimension])
        
        y = roi_pooling(input, rois, pool_height=pooled_w, pool_width=pooled_h)
        
        self.assertEqual(y.get_shape().ndims, 4)
        self.assertIs(y.get_shape()[0].value, n_rois)
        self.assertIs(y.get_shape()[1].value, n_channels)
        self.assertIs(y.get_shape()[2].value, pooled_h)
        self.assertIs(y.get_shape()[3].value, pooled_w)

if __name__ == '__main__':
    tf.test.main()
