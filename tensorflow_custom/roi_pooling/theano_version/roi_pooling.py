import theano
import theano.tensor as T

from theano import Apply
from theano.gof import COp
from theano.gradient import grad_undefined
from theano.sandbox.cuda import as_cuda_ndarray_variable, GpuOp

class ROIPoolingOp(GpuOp, COp):
  __props__ = ('pooled_h', 'pooled_w', 'spatial_scale')

  func_file = "./roi_pooling.cu"
  func_name = "APPLY_SPECIFIC(Forward_gpu)"

  def __init__(self, pooled_h, pooled_w, spatial_scale):
    super(ROIPoolingOp, self).__init__(self.func_file,
                                       self.func_name)

    self.pooled_h = pooled_h
    self.pooled_w = pooled_w
    self.spatial_scale = spatial_scale

  def make_node(self, data, rois):
    data = as_cuda_ndarray_variable(data)
    rois = as_cuda_ndarray_variable(rois)
    assert data.ndim == 4
    assert rois.ndim == 2

    return Apply(self, [data, rois], [data.type(), data.type()])

  def get_op_params(self):
    return [('POOLED_HEIGHT', str(self.pooled_h)),
            ('POOLED_WIDTH', str(self.pooled_w)),
            ('SPATIAL_SCALE', str(self.spatial_scale))]

  def infer_shape(self, node, in_shapes):
    data_shape = T.shape(node.inputs[0])
    rois_shape = T.shape(node.inputs[1])
    batch_size = rois_shape[0]
    num_maps = data_shape[1]
    h = self.pooled_h
    w = self.pooled_w
    out_shape = [batch_size, num_maps, h, w]
    return [out_shape, out_shape]

  def grad(self, inp, grads):
    outs = self(*inp)
    grad_op = ROIPoolingGradOp(
        self.pooled_h, self.pooled_w, self.spatial_scale)
    data_grad = grad_op(*(inp + [outs[1], grads[0]]))

    return [data_grad, grad_undefined(self, 1, inp[1])]

  def __eq__(self, other):
    return (type(self) == type(other) and
            self.pooled_h == other.pooled_h and
            self.pooled_w == other.pooled_w and
            self.spatial_scale == other.spatial_scale)

  def __hash__(self):
    return (hash(type(self)) ^ 
            hash(self.pooled_h) ^
            hash(self.pooled_w) ^
            hash(self.spatial_scale)) 

  def c_code_cache_version(self):
    return (1,)

class ROIPoolingGradOp(GpuOp, COp):
  __props__ = ('pooled_h', 'pooled_w', 'spatial_scale')

  func_file = "./roi_pooling.cu"
  func_name = "APPLY_SPECIFIC(Backward_gpu)"

  def __init__(self, pooled_h, pooled_w, spatial_scale):
    super(ROIPoolingGradOp, self).__init__(self.func_file,
                                           self.func_name)

    self.pooled_h = pooled_h
    self.pooled_w = pooled_w
    self.spatial_scale = spatial_scale

  def make_node(self, data, rois, argmaxes, out_grad):
    data = as_cuda_ndarray_variable(data)
    rois = as_cuda_ndarray_variable(rois)
    argmaxes = as_cuda_ndarray_variable(argmaxes)
    out_grad = as_cuda_ndarray_variable(out_grad)
    assert data.ndim == 4
    assert rois.ndim == 2
    assert argmaxes.ndim == 4
    assert out_grad.ndim == 4

    return Apply(self, [data, rois, argmaxes, out_grad], [data.type()])

  def get_op_params(self):
    return [('POOLED_HEIGHT', str(self.pooled_h)),
            ('POOLED_WIDTH', str(self.pooled_w)),
            ('SPATIAL_SCALE', str(self.spatial_scale))]

  def infer_shape(self, node, in_shapes):
    return [in_shapes[0]]

  def grad(self, inp, grads):
    return [grad_undefined(self, i, inp[i]) for i in xrange(4)]

  def __eq__(self, other):
    return (type(self) == type(other) and
            self.pooled_h == other.pooled_h and
            self.pooled_w == other.pooled_w and
            self.spatial_scale == other.spatial_scale)

  def __hash__(self):
    return (hash(type(self)) ^ 
            hash(self.pooled_h) ^
            hash(self.pooled_w) ^
            hash(self.spatial_scale)) 

  def c_code_cache_version(self):
    return (1,)
