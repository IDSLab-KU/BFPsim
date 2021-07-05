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

from blockfunc import make_groups_tensor, make_groups_tensor_fc
from functions import BFConf

DEFAULT_CUDA = True

# BlockFloat Linear Function
class BFLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, bf_conf):    
        # Grouping input and weight
        if bf_conf.fi:
            input = make_groups_tensor_fc(input, bf_conf.fi_bit, bf_conf.fi_sz, bf_conf.fi_dir)
        if bf_conf.fw:
            weight = make_groups_tensor_fc(weight, bf_conf.fw_bit, bf_conf.fw_sz, bf_conf.fw_dir)
        # Save context to use on backward
        ctx.bf_conf = bf_conf
        ctx.save_for_backward(input, weight, bias)
        
        # Compute FC and return
        output = F.linear(input, weight, bias)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Grouping Output
        if bf_conf.fo:
            output = make_groups_tensor_fc(output, bf_conf.fo_bit, bf_conf.fo_sz, bf_conf.fo_dir)
        print("F: %s %s %s"%(input.shape, weight.shape, output.shape))
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        # input, weight, bias, confs = ctx.saved_tensors
        input, weight, bias = ctx.saved_tensors
        bf_conf = ctx.bf_conf
        print("output gradient")

        # Calculate gradients
        grad_input = grad_weight = grad_bias = None

        # Calculate Input Gradient
        ## Grouping grad_output
        if bf_conf.bio:
            grad_output_ = make_groups_tensor_fc(grad_output, bf_conf.bio_bit, bf_conf.bio_sz, bf_conf.bio_dir)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping weight
        if bf_conf.biw:
            weight = make_groups_tensor_fc(weight, bf_conf.biw_bit, bf_conf.biw_sz, bf_conf.biw_dir)        

        print(grad_output_.shape)
        print(weight.shape)
        grad_input_ = F.linear(grad_output_, weight.t(), bias).t()
        print(grad_input.shape)
        # grad_input_ = grad_output_.mm(weight)

        if bf_conf.big:
            grad_input_ = make_groups_tensor_fc(grad_input, bf_conf.big_bit, bf_conf.big_sz, bf_conf.big_dir)
        else: # If not grouping, use original type
            grad_input_ = grad_input

        if bf_conf.bwo:
            # Regroup if bwo / bio grouping configuration is different!
            if (bf_conf.bwo_bit != bf_conf.bio_bit or bf_conf.bwo_sz != bf_conf.bio_sz or bf_conf.bwo_dir != bf_conf.bio_dir):
                grad_output_ = make_groups_tensor_fc(grad_output, bf_conf.bwo_bit, bf_conf.bwo_sz, bf_conf.bwo_dir)
        else: # If not grouping, use original type
            grad_output_ = grad_output
        ## Grouping input - it's not grad_input, right?
        if bf_conf.bwi:
            # Regroup if bwi / fi grouping configuration is different!
            if (bf_conf.bwi_bit != bf_conf.fi_bit or bf_conf.bwi_sz != bf_conf.fi_sz or bf_conf.bwi_dir != bf_conf.fi_dir):
                input = make_groups_tensor_fc(input, bf_conf.bwi_bit, bf_conf.bwi_sz, bf_conf.bwi_dir)
        grad_weight = F.linear(grad_output_.t(), input.t(), bias).t()
        # grad_weight = grad_output_.t().mm(input)
        # print(grad_weight.shape)
        # Group the gradient of weight
        if bf_conf.bwg:
            grad_weight = make_groups_tensor_fc(grad_weight, bf_conf.bwg_bit, bf_conf.bwg_sz, bf_conf.bwg_dir)

        if bf_conf.bwg_boost != 1.0:
            grad_weight /= bf_conf.bwg_boost

        if bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input_, grad_weight, grad_bias, None

# Blockfloat Linear
class BFLinear(torch.nn.Module):
    def __init__(self,
                input_features: int,
                output_features: int,
                bf_conf: BFConf, bias=True):
        super(BFLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        self.bf_conf = bf_conf

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
        return BFLinearFunction.apply(input, self.weight, self.bias, self.bf_conf)
    
    def extra_repr(self):
        s = ('{input_features}, {output_features}')
        s += ', bf_conf=({bf_conf})'
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
    def forward(ctx, input, weight, bias=None, bf_conf=None, stride=1, padding=0, dilation=1, groups=1):
        # print("= Forward:",input.shape, weight.shape, stride, padding, dilation, groups)
        # Grouping input and weight
        if bf_conf.fi:
            input = make_groups_tensor(input, bf_conf.fi_bit, bf_conf.fi_sz, bf_conf.fi_dir, 0)
        if bf_conf.fw:
            weight = make_groups_tensor(weight, bf_conf.fw_bit, bf_conf.fw_sz, bf_conf.fw_dir, 1)

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bf_conf = bf_conf

        ctx.save_for_backward(input, weight, bias)

        # Compute Convolution
        output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        # Grouping Output
        if bf_conf.fo:
            output = make_groups_tensor(output, bf_conf.fo_bit, bf_conf.fo_sz, bf_conf.fo_dir, 2)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and configs
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        bf_conf = ctx.bf_conf

        # print("= Backward:",grad_output.shape, stride, padding, dilation, groups)
        grad_input = grad_weight = grad_bias = None

        # Calculate Input Gradient
        ## Grouping grad_output
        if bf_conf.bio:
            grad_output_ = make_groups_tensor(grad_output, bf_conf.bio_bit, bf_conf.bio_sz, bf_conf.bio_dir, 10)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping weight
        if bf_conf.biw:
            weight = make_groups_tensor(weight, bf_conf.biw_bit, bf_conf.biw_sz, bf_conf.biw_dir, 11)        
        ## Do the convolution
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output_, stride, padding, dilation, groups)
        ## Grouping output grad_input
        if bf_conf.big:
            grad_input_ = make_groups_tensor(grad_input, bf_conf.big_bit, bf_conf.big_sz, bf_conf.big_dir, 12)
        else: # If not grouping, use original type
            grad_input_ = grad_input

        # Calculate Weight Gradient (2D Convolution, Depthwise Convolution)
        ## Grouping grad_output
        
        if bf_conf.bwo:
            # Regroup if bwo / bio grouping configuration is different!
            if (bf_conf.bwo_bit != bf_conf.bio_bit or bf_conf.bwo_sz != bf_conf.bio_sz or bf_conf.bwo_dir != bf_conf.bio_dir):
                grad_output_ = make_groups_tensor(grad_output, bf_conf.bwo_bit, bf_conf.bwo_sz, bf_conf.bwo_dir, 20)
        else: # If not grouping, use original type
            grad_output_ = grad_output
        ## Grouping input - it's not grad_input, right?
        if bf_conf.bwi:
            # Regroup if bwi / fi grouping configuration is different!
            if (bf_conf.bwi_bit != bf_conf.fi_bit or bf_conf.bwi_sz != bf_conf.fi_sz or bf_conf.bwi_dir != bf_conf.fi_dir):
                input = make_groups_tensor(input, bf_conf.bwi_bit, bf_conf.bwi_sz, bf_conf.bwi_dir, 21)
        ## Do the convolution
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output_, stride, padding, dilation, groups)
        # Group the gradient of weight
        if bf_conf.bwg:
            grad_weight = make_groups_tensor(grad_weight, bf_conf.bwg_bit, bf_conf.bwg_sz, bf_conf.bwg_dir, 22)

        # Apply weaken gradient if weight gradient boost is applied
        if bf_conf.bwg_boost != 1.0:
            grad_weight /= bf_conf.bwg_boost

        # TODO : Fix Bias Grouping
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0,2,3)).squeeze(0)
        
        return grad_input_, grad_weight, grad_bias, None, None, None, None, None

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
        # TODO : Edit this area
        if type(kernel_size) == int:
            self.kernel_size = kernel_size
        else:
            self.kernel_size = kernel_size[0]

        self.bf_conf = bf_conf
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.padding_mode = padding_mode

        self.weight = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, self.kernel_size, self.kernel_size))
        # self.bias = nn.Parameter(torch.Tensor(out_channels))
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
        return BFConv2dFunction.apply(input, self.weight, self.bias, self.bf_conf, self.stride, self.padding, self.dilation, self.groups)
    
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
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
