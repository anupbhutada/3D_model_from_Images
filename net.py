import numpy as np
import datetime as dt

import torch

class Net(Module):
    
    def __init__(self, random_seed=dt.datetime.now().microsecond, compute_grad=True):
        super(Net, self).__init__()
        self.batch_size = 36
        self.img_w = 127
        self.img_h = 127
        self.n_vox = 32
        self.compute_grad = compute_grad
        '''
        # (self.batch_size, 3, self.img_h, self.img_w),
        # override x and is_x_tensor4 when using multi-view network
        self.x = tensor.tensor4()
        self.is_x_tensor4 = True

        # (self.batch_size, self.n_vox, 2, self.n_vox, self.n_vox),
        self.y = tensor5()
        '''
        self.activations = []  # list of all intermediate activations
        self.loss = []  # final loss
        self.output = []  # final output
        self.error = []  # final output error
        self.params = []  # all learnable params
        self.grads = []  # will be filled out automatically

    def network_definition(self, x):
        pass
    def forward(self, x):
        return self.network_definition(x)
    
    def __str__(self):
        return "GRUNet"