import torch
import torch.nn as nn
import torch.nn.functional as F

from functions import make_groups_tensor

import math


# Temp relu, can be removed since relu doesn't need to be optimized
class BlockReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input

CONF_BIT, CONF_GROUP_SIZE = 8, 36
CONF_CUDA = True

# BlockFloat Linear Function
class BFLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):

        # Setting NEW method! (cuda...)
        if CONF_CUDA:
            input = make_groups_tensor(input.cpu(), CONF_BIT, group_size = CONF_GROUP_SIZE).cuda()
            weight = make_groups_tensor(weight.cpu(), CONF_BIT, group_size = CONF_GROUP_SIZE).cuda()
        else:
            input = make_groups_tensor(input, CONF_BIT, group_size = CONF_GROUP_SIZE)
            weight = make_groups_tensor(weight, CONF_BIT, group_size = CONF_GROUP_SIZE)

        # changing save position...?
        ctx.save_for_backward(input, weight, bias)
        
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        if bias is not None:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, grad_bias


import numpy as np

from typing import TypeVar, Union, Tuple, Optional
T = TypeVar('T')
_scalar_or_tuple_any_t = Union[T, Tuple[T, ...]]
_scalar_or_tuple_1_t = Union[T, Tuple[T]]
_scalar_or_tuple_2_t = Union[T, Tuple[T, T]]
_scalar_or_tuple_3_t = Union[T, Tuple[T, T, T]]

_size_1_t = _scalar_or_tuple_1_t[int]
_size_2_t = _scalar_or_tuple_2_t[int]
_size_3_t = _scalar_or_tuple_3_t[int]

# Blockfloat Linear
class BFLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(BFLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

        # Weight parameters, should be grouped with few numbers
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_paramter('bias', None)
        
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return BFLinearFunction.apply(input, self.weight, self.bias)
    
    def extra_repr(self):
        # Extra information about this module
        return 'input_features={}, output_features={}, bias={}'.format(
            self.input_features, self.output_features, self.bias is not None
        )

# Blockfloat Convolution Function
# TODO : Implement Conv2d Operation
# https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7
class BFConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1):
        ctx.save_for_backward(input, weight, bias)
        # print("Before: ",input.shape, weight.shape)
        if CONF_CUDA:
            input = make_groups_tensor(input.cpu(), CONF_BIT, group_size = CONF_GROUP_SIZE).cuda()
            weight = make_groups_tensor(weight.cpu(), CONF_BIT, group_size = CONF_GROUP_SIZE).cuda()
        else:
            input = make_groups_tensor(input, CONF_BIT, group_size = CONF_GROUP_SIZE)
            weight = make_groups_tensor(weight, CONF_BIT, group_size = CONF_GROUP_SIZE)
        # print("After: ",input.shape, weight.shape)
        return F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, w, b = ctx.saved_variables
        x_grad = w_grad = None
        if ctx.needs_input_grad[0]:
            x_grad = torch.nn.grad.conv2d_input(x.shape, w, grad_output)
        if ctx.needs_input_grad[1]:
            w_grad = torch.nn.grad.conv2d_weight(x, w.shape, grad_output)
        return x_grad, w_grad
"""
https://discuss.pytorch.org/t/how-to-find-the-source-code-of-conv2d-backward-function/19139/9
    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_variables
            
        grad_input = grad_weight= grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output)
            
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output)
                
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        if bias is not True:
            return grad_input, grad_weight, grad_bias
        else:
            return grad_input, grad_weight
"""

# Blockfloat Convolution
class BFConv2d(torch.nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
    #     padding=0, groups=1, bias=False, padding_mode='zeros'):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: _size_2_t,
                stride: _size_2_t = 1,
                padding: _size_2_t = 0,
                dilation: _size_2_t = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros'):
        super(BFConv2d, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        """
        self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, self.groups = in_channels, out_channels, kernel_size,stride, padding, groups
        # From torch/nn/modules/conv.py
        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        """
    
    def reset_parameters(self) -> None:
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        # Please edit here, too!!
        # return BFConv2dFunction.apply(input, self.weight, self.bias, self.stride, self.padding, self.groups)
        return BFConv2dFunction.apply(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def extra_repr(self):
        # From /torch/nn/modules/conv.py
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        s += ', padding={padding}'
        s += ', dilation={dilation}'
        s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
