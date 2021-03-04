import torch
import torch.nn as nn
import torch.nn.functional as F

import math
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

from functions import make_groups_tensor, BFConf

DEFAULT_GROUP_MANTISSA = 8
DEFAULT_GROUP_SIZE = 36
DEFAULT_GROUP_DIRECTION = None
DEFAULT_CUDA = True


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

# BlockFloat Linear Function
class BFLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bf_conf, bias, cuda=DEFAULT_CUDA):    
        if cuda:
            input = make_groups_tensor(input.cpu(), bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir).cuda()
            weight = make_groups_tensor(weight.cpu(), bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir).cuda()
        else:
            input = make_groups_tensor(input, bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir)
            weight = make_groups_tensor(weight, bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir)

        # Save context to use on backward
        cuda = 1 if cuda else 0
        confs = torch.from_numpy(np.array([cuda]))
        ctx.save_for_backward(input, weight, bias, confs)
        
        # Compute FC and return
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, weight, bias, confs = ctx.saved_tensors
        confs = confs.numpy()
        # group_mantissa, group_size, group_direction, cuda = confs[0], confs[1], confs[2], confs[3]
        cuda = confs[0]
        cuda = True if cuda > 1 else False

        # Calculate gradients
        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)
        if bias is not None:
            grad_bias = grad_output.sum(0)
        
        return grad_input, grad_weight, None, grad_bias, None

# Blockfloat Linear
class BFLinear(torch.nn.Module):
    def __init__(self,
                input_features: int,
                output_features: int,
                bf_conf: BFConf,
                bias: bool = True,
                cuda: bool = DEFAULT_CUDA):
        super(BFLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bf_conf = bf_conf
        self.cuda = cuda

        # Weight parameters, should be grouped with few numbers
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            self.register_paramter('bias', None)
        
        # Initialize weights manually
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return BFLinearFunction.apply(input, self.weight, self.bf_conf, self.bias, self.cuda)
    
    def extra_repr(self):
        s = ('{input_features}, {output_features}')
        s += ', bf_conf=({bf_conf})'
        if self.cuda is False:
            s += ', cuda=False'
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        return s.format(**self.__dict__)


# Blockfloat Convolution Function
# TODO : Implement Conv2d Operation
# https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7
class BFConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bf_conf, bias=None, stride=1, padding=0, dilation=1, groups=1, cuda=DEFAULT_CUDA):
        # print("= Forward:",input.shape, weight.shape, stride, padding, dilation, groups)
        # Block input and weight's values to unify mantissa
        if cuda:
            input = make_groups_tensor(input.cpu(), bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir).cuda()
            weight = make_groups_tensor(weight.cpu(), bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir).cuda()
        else:
            input = make_groups_tensor(input, bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir)
            weight = make_groups_tensor(weight, bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir)

        # Save arguments to context to use on backward
        # WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
        # if group_direction == None:
        #     group_direction = 0
        cuda = 1 if cuda else 0
        confs = torch.from_numpy(np.array([stride, padding, dilation, groups, cuda]))
        ctx.save_for_backward(input, weight, bias, confs)

        # Compute Convolution
        return F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, weight, bias, confs = ctx.saved_variables
        confs = confs.numpy()
        # stride, padding, dilation, groups, group_mantissa, group_size, group_direction, cuda = confs[0], confs[1], confs[2], confs[3], confs[4], confs[5], confs[6], confs[7]
        stride, padding, dilation, groups, cuda = confs[0], confs[1], confs[2], confs[3], confs[4]

        # print("= Backward:",grad_output.shape, stride, padding, dilation, groups)
        cuda = True if cuda > 0 else False
        
        # Gradient Grouping
        if cuda:
            grad_output = make_groups_tensor(grad_output.cpu(), 8, group_size = 36, group_direction = (2,3,0,1)).cuda()
        else:
            grad_output = make_groups_tensor(grad_output, 8, group_size = 36, group_direction = (2,3,0,1))

        # Calculate Gradient
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)           
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)
        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)
        return grad_input, grad_weight, None, grad_bias, None, None, None, None, None

# Blockfloat Convolution
class BFConv2d(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: _size_2_t,
                bf_conf: BFConf, 
                stride: _size_2_t = 1,
                padding: _size_2_t = 0,
                dilation: _size_2_t = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros',
                cuda=DEFAULT_CUDA):
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
        self.bf_conf = bf_conf
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode
        self.cuda = cuda

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()


    def reset_parameters(self) -> None:
        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_uniform_
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return BFConv2dFunction.apply(input, self.weight, self.bf_conf,
                self.bias, self.stride, self.padding, self.dilation,
                self.groups, self.cuda)
    def extra_repr(self):
        # From /torch/nn/modules/conv.py
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != 0:
            s += ', padding={padding}'
        if self.dilation != 1:
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        s += ', bf_conf=({bf_conf})'
        if self.cuda is False:
            s += ', cuda=False'
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)