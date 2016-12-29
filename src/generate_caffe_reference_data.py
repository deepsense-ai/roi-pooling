import tempfile
import os

import caffe
import numpy as np

GPU_ID = 0 # Switch between 0 and 1 depending on the GPU you want to use.

input_image_file = 'test_input.npy'
input_roi_file = 'test_roi.npy'
caffe_output_file = 'test_caffe_output.npy'
caffe_grad_file =  'test_caffe_grad.npy'
filenames = [input_image_file, input_roi_file, caffe_output_file, caffe_grad_file]


def roinet_file(width, height, pooled_w, pooled_h, n_rois):
    with tempfile.NamedTemporaryFile(delete=False) as f:
        f.write("""
        name: 'roi pooling net' force_backward: true
        input: 'data' input_shape {{ dim: 1 dim: 1 dim: {width} dim: {height} }}
        
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
          }}""".format(width=width, height=height, pooled_w=pooled_w, pooled_h=pooled_h, n_rois=n_rois))
        return f.name


def ROI_caffe(x, rois, pooled_w, pooled_h):
    width = x.shape[2]
    height = x.shape[3]
    net_file = roinet_file(width, height, pooled_w=pooled_w, pooled_h=pooled_h, n_rois=len(rois))
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


def load_data():
    return [np.load(f) for f in filenames]


def generate_image_and_rois(width, height, n_rois):
    x = np.random.uniform(low=0, high=2048, size=(1,1,width, height))
    x = np.floor(x)

    x1 = np.random.uniform(low=0, high=width, size=(n_rois))
    x1 = np.floor(x1)
    y1 = np.random.uniform(low=0, high=height, size=(n_rois))
    y1 = np.floor(y1)

    roi_widths = np.random.uniform(low=0, high=width, size=(n_rois))
    roi_widths = np.floor(roi_widths)
    roi_heights = np.random.uniform(low=0, high=height, size=(n_rois))
    roi_heights = np.floor(roi_heights)

     # crop the regions
    roi_widths = np.where(x1 + roi_widths > width - 1, width - x1, roi_widths)
    roi_heights = np.where(y1 + roi_heights > height - 1, height - y1, roi_heights)

    x2 = x1 + roi_widths
    y2 = y1 + roi_heights
    channels = np.zeros_like(x1)
    rois = np.vstack([channels, x1, y1, x2, y2]).T
    return x, rois


if __name__ == '__main__':
    print 'Generating reference ROI layer outputs with caffe implementation on GPU{}...'.format(GPU_ID)
    caffe.set_mode_gpu()
    caffe.set_device(GPU_ID)

    x, rois = generate_image_and_rois(width=1000, height=1000, n_rois=1000)
    caffe_y, caffe_grad = ROI_caffe(x, rois, 7, 7)

    arrays_to_save = [
        np.transpose(x, axes=(0, 2, 3, 1)),
        rois,
        caffe_y,
        np.transpose(caffe_grad, axes=(0, 2, 3, 1))
    ]

    print 'Saving files...'
    for filename, array in zip(filenames, arrays_to_save):
        np.save(filename, array)
    print('Done!')