import torch
from torch.nn import Module
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
import numpy as np
dtype = torch.cuda.FloatTensor

class InputLayer(object):

    def __init__(self, input_shape, tinput=None):
        self._output_shape = input_shape
        self._input = tinput

    @property
    def output(self):
        if self._input is None:
            raise ValueError('Cannot call output for the layer. Initialize' \
                             + ' the layer with an input argument')
        return self._input

    @property
    def output_shape(self):
        return self._output_shape


class SoftmaxWithLoss3D(object):
    """
    Softmax with loss (n_batch, n_vox, n_label, n_vox, n_vox)
    """

    def __init__(self, input):
        self.input = input
        self.exp_x = torch.exp(self.input)
        self.sum_exp_x = torch.sum(self.exp_x, axis=2, keepdims=True)

    def prediction(self):
        return self.exp_x / self.sum_exp_x

    def error(self, y, threshold=0.5):
        return tensor.mean(tensor.eq(tensor.ge(self.prediction(), threshold), y))

    def loss(self, y):
        """
        y must be a tensor that has the same dimensions as the input. For each
        channel, only one element is one indicating the ground truth prediction
        label.
        """
        return tensor.mean(
            tensor.sum(-y * self.input, axis=2, keepdims=True) + tensor.log(self.sum_exp_x))

    
class Layer(object):
    def __init__(self, prev_layer):
        self._output = None
        self._output_shape = None
        self._prev_layer = prev_layer
        self._input_shape = prev_layer.output_shape
        
    def set_output(self):
        '''Override the function'''
        # set self._output using self._input=self._prev_layer.output
        raise NotImplementedError('Layer virtual class')

    @property
    def output_shape(self):
        if self._output_shape is None:
            raise ValueError('Set output shape first')
        return self._output_shape

    @property
    def output(self):
        if self._output is None:
            self.set_output()
        return self._output
        
        
class ConvLayer(Layer):
    def __init__(self, prev_layer, filter_shape, padding=True, params=None):
        super(ConvLayer, self).__init__(prev_layer)
        self._padding = padding
        self._filter_shape = [filter_shape[0], self._input_shape[1], filter_shape[1],
                              filter_shape[2]]
        
        if params is None:
            self.W = Variable(torch.randn(self._filter_shape).type(dtype), requires_grad=True)
            self.b = Variable(torch.randn(self._filter_shape[0],).type(dtype), requires_grad=True)
        else:
            self.W = params[0]
            self.b = params[1]
            
        self.params = [self.W, self.b]
        
        if padding and filter_shape[1] * filter_shape[2] > 1:
            self._padding = [0, 0, int((filter_shape[1] - 1) / 2), int((filter_shape[2] - 1) / 2)]
            self._output_shape = [self._input_shape[0], filter_shape[0], self._input_shape[2],
                                  self._input_shape[3]]
        else:
            self._padding = [0] * 4
            # TODO: for the 'valid' convolution mode the following is the
            # output shape. Diagnose failure
            self._output_shape = [self._input_shape[0], filter_shape[0],
                                  self._input_shape[2] - filter_shape[1] + 1,
                                  self._input_shape[3] - filter_shape[2] + 1]
        
    def set_output(self):
        prev_output = self._prev_layer.output
        padding = self._padding
            
        if sum(self._padding) > 0:
            padder = nn.ZeroPad2d((padding[3], padding[3],
                                 padding[2], padding[2],
                                 0, 0,
                                 0, 0))
            padded_input = padder(prev_output)
            padded_input_shape = [self._input_shape[0], self._input_shape[1],
                              self._input_shape[2] + 2 * self._padding[2],
                              self._input_shape[3] + 2 * self._padding[3]]

        else:
            padded_input = prev_output
            padded_input_shape = self._input_shape

        conv_out = F.conv2d(padded_input, self.W, bias=self.b, padding=self._padding)

        self._output = conv_out
            
class PoolLayer(Layer):

    def __init__(self, prev_layer, pool_size=(2, 2), padding=(1, 1)):
        super(PoolLayer, self).__init__(prev_layer)
        self._pool_size = pool_size
        self._padding = padding
        img_rows = self._input_shape[2] + 2 * padding[0]
        img_cols = self._input_shape[3] + 2 * padding[1]
        out_r = (img_rows - pool_size[0]) // pool_size[0] + 1
        out_c = (img_cols - pool_size[1]) // pool_size[1] + 1
        self._output_shape = [self._input_shape[0], self._input_shape[1], out_r, out_c]

    def set_output(self):
        pooled_out = F.max_pool2d(self._prev_layer.output, self._pool_size, padding=self._padding)
        self._output = pooled_out
        
class FlattenLayer(Layer):

    def __init__(self, prev_layer):
        super(FlattenLayer, self).__init__(prev_layer)
        self._output_shape = [self._input_shape[0], np.prod(self._input_shape[1:])]

    def set_output(self):
        self._output = self._prev_layer.output.view(self._input_shape[0], -1)  # flatten from the second dim

            

class TensorProductLayer(Layer):

    def __init__(self, prev_layer, n_out, params=None, bias=True):
        super(TensorProductLayer, self).__init__(prev_layer)
        self._bias = bias
        n_in = self._input_shape[-1]

        if params is None:
            self.W = Variable(torch.randn(n_in, n_out), requires_grad=True)
            if bias:
                self.b = Variable(torch.randn(n_out,) + 0.1, requires_grad=True)
        else:
            self.W = params[0]
            if bias:
                self.b = params[1]

        # parameters of the model
        self.params = [self.W]
        if bias:
            self.params.append(self.b)

        self._output_shape = [self._input_shape[0]]
        self._output_shape.extend(self._input_shape[1:-1])
        self._output_shape.append(n_out)

    def set_output(self):
        self._output = torch.mm(self._prev_layer.output, self.W)
        if self._bias:
            self._output += self.b

            
class Conv3DLayer(Layer):
    """3D Convolution layer"""

    def __init__(self, prev_layer, filter_shape, padding=None, params=None):
        super(Conv3DLayer, self).__init__(prev_layer)
        self._filter_shape = [filter_shape[0],  # out channel
                              self._input_shape[2],  # in channel
                              filter_shape[1],  # time
                              filter_shape[2],  # height
                              filter_shape[3]]  # width
        self._padding = padding

        if params is None:
            self.W = Variable(torch.randn(self._filter_shape), requires_grad=True)
            self.b = Variable(torch.randn(self.filter_shape[0]), requires_grad=True)
            params = [self.W, self.b]
        else:
            self.W = params[0]
            self.b = params[1]

        self.params = [self.W, self.b]

        if padding is None:
            self._padding = [0, int((filter_shape[1] - 1) / 2), 0, int((filter_shape[2] - 1) / 2),
                             int((filter_shape[3] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        if np.sum(self._padding) > 0:
            
            padder = nn.ZeroPad2d((padding[4], padding[4],
                                padding[3], padding[3],
                                0, 0,
                                padding[1], padding[1],
                                0, 0))
            
            padded_input = padder(self._prev_layer.output)
            
        else:
            padded_input = self._prev_layer.output

        self._output = F.conv3d(padded_input, self.W) + \
            self.b.view(1, 1, self.b.shape[0], 1, 1)

            
class FCConv3DLayer(Layer):
    """3D Convolution layer with FC input and hidden unit"""

    def __init__(self, prev_layer, fc_layer, filter_shape, padding=None, params=None):
        """Prev layer is the 3D hidden layer"""
        super(FCConv3DLayer, self).__init__(prev_layer)
        self._fc_layer = fc_layer
        self._filter_shape = [filter_shape[0],  # out channel
                              filter_shape[1],  # in channel
                              filter_shape[2],  # time
                              filter_shape[3],  # height
                              filter_shape[4]]  # width
        self._padding = padding

        if padding is None:
            self._padding = [0, int((self._filter_shape[1] - 1) / 2), 0, int(
                (self._filter_shape[3] - 1) / 2), int((self._filter_shape[4] - 1) / 2)]

        self._output_shape = [self._input_shape[0], self._input_shape[1], filter_shape[0],
                              self._input_shape[3], self._input_shape[4]]

        if params is None:
            self.Wh = Variable(torch.randn(self._filter_shape), requires_grad=True)

            self._Wx_shape = [self._fc_layer._output_shape[1], np.prod(self._output_shape[1:])]

            # Each 3D cell will have independent weights but for computational
            # speed, we expand the cells and compute a matrix multiplication.
            self.Wx = Variable(torch.randn(self._Wx_shape), requires_grad=True)

            self.b = Variable(torch.randn(filter_shape[0],) + 0.1, requires_grad=True)
            params = [self.Wh, self.Wx, self.b]
        else:
            self.Wh = params[0]
            self.Wx = params[1]
            self.b = params[2]

        self.params = [self.Wh, self.Wx, self.b]

    def set_output(self):
        padding = self._padding
        input_shape = self._input_shape
        
        padder = nn.ZeroPad2d((padding[4], padding[4],
                               padding[3], padding[3],
                               0, 0,
                               padding[1], padding[1],
                               0, 0))
        padded_input = padder(self._prev_layer.output)
        
        fc_output = torch.matmul(self._fc_layer.output, self.Wx).view(self._output_shape)
        self._output = F.conv3d(padded_input, self.Wh) + fc_output + self.b.view(1, 1, -1, 1, 1)
            
        '''
        fc_output = tensor.reshape(
            tensor.dot(self._fc_layer.output, self.Wx.val), self._output_shape)
        self._output = conv3d2d.conv3d(padded_input, self.Wh.val) + \
            fc_output + self.b.val.dimshuffle('x', 'x', 0, 'x', 'x')
            '''
        

class Unpool3DLayer(Layer):
    """3D Unpooling layer for a convolutional network """

    def __init__(self, prev_layer, unpool_size=(2, 2, 2), padding=(0, 0, 0)):
        super(Unpool3DLayer, self).__init__(prev_layer)
        self._unpool_size = unpool_size
        self._padding = padding
        output_shape = (self._input_shape[0],  # batch
                        unpool_size[0] * self._input_shape[1] + 2 * padding[0],  # depth
                        self._input_shape[2],  # out channel
                        unpool_size[1] * self._input_shape[3] + 2 * padding[1],  # row
                        unpool_size[2] * self._input_shape[4] + 2 * padding[2])  # col
        self._output_shape = output_shape

    def set_output(self):
        output_shape = self._output_shape
        padding = self._padding
        unpool_size = self._unpool_size
        
        padder = nn.ZeroPad2d((padding[2], padding[2],
                               padding[1], padding[1],
                               0, 0,
                               padding[0], padding[0],
                               0, 0))
        
        unpooled_output = padder(self._prev_layer.output)
        
        self._output = unpooled_output


            
class LeakyReLU(Layer):

    def __init__(self, prev_layer, leakiness=0.01):
        super(LeakyReLU, self).__init__(prev_layer)
        self._leakiness = leakiness
        self._output_shape = self._input_shape

    def set_output(self):
        self._input = self._prev_layer.output
        if self._leakiness:
            f1 = 0.5 * (1 + self._leakiness)
            f2 = 0.5 * (1 - self._leakiness)
            self._output = f1 * self._input + f2 * abs(self._input)
            # self.param = [self.leakiness]
        else:
            self._output = 0.5 * (self._input + abs(self._input))
            
            
class AddLayer(Layer):

    def __init__(self, prev_layer, add_layer):
        super(AddLayer, self).__init__(prev_layer)
        self._output_shape = self._input_shape
        self._add_layer = add_layer

    def set_output(self):
        self._output = self._prev_layer.output + self._add_layer.output


class EltwiseMultiplyLayer(Layer):

    def __init__(self, prev_layer, mult_layer):
        super(EltwiseMultiplyLayer, self).__init__(prev_layer)
        self._output_shape = self._input_shape
        self._mult_layer = mult_layer

    def set_output(self):
        self._output = self._prev_layer.output * self._mult_layer.output
        

class SigmoidLayer(Layer):

    def __init__(self, prev_layer):
        super(SigmoidLayer, self).__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = torch.sigmoid(self._prev_layer.output)
        

class TanhLayer(Layer):

    def __init__(self, prev_layer):
        super(TanhLayer, self).__init__(prev_layer)

    def set_output(self):
        self._output = torch.tanh(self._prev_layer.output)


class ComplementLayer(Layer):
    """ Compute 1 - input_layer.output """

    def __init__(self, prev_layer):
        super(ComplementLayer, self).__init__(prev_layer)
        self._output_shape = self._input_shape

    def set_output(self):
        self._output = torch.ones(self._prev_layer.output.shape) - self._prev_layer.output

#***********************************
#check whether the shapes for input and filters are compatible with torch functions for conv layers