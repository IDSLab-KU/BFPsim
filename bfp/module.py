"""
    This code is part of the BFPSim (https://github.com/ids-Lab-DGIST/BFPSim)

    Seunghyun Lee (R3C0D3r) from IDSLab, DGIST
    coder@dgist.ac.kr

    License: CC BY 4.0
"""

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

from bfp.internal import make_groups_tensor, gradient_linear_weight_2d, gradient_linear_weight_3d
from bfp.conf import BFPConf

# BlockFloat Linear Function
class BFPLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, bfp_conf=None):
        # print("input:", input.shape)
        # Grouping input and weight
        
        if bfp_conf.fi:
            input_ = make_groups_tensor(input.clone().detach(), bfp_conf.fi_bit, bfp_conf.fi_dim)
        if bfp_conf.fw:
            weight_ = make_groups_tensor(weight.clone().detach(), bfp_conf.fw_bit, bfp_conf.fw_dim)

        # Save context to use on backward
        ctx.bfp_conf = bfp_conf
        ctx.save_for_backward(input_, weight_, bias)
        
        if bias != None:
            output = F.linear(input_, weight_, bias)
        else:
            output = F.linear(input_, weight_)
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Grouping Output
        if bfp_conf.fo:
            output = make_groups_tensor(output, bfp_conf.fo_bit, bfp_conf.fo_dim)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        input, weight, bias = ctx.saved_tensors
        bfp_conf = ctx.bfp_conf

        # Calculate gradients
        grad_input = grad_weight = grad_bias = None

        # Calculate Input Gradient
        ## Grouping grad_output
        if bfp_conf.bio:
            grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bio_bit, bfp_conf.bio_dim)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping weight
        if bfp_conf.biw:
            weight_ = make_groups_tensor(weight.clone().detach(), bfp_conf.biw_bit, bfp_conf.biw_dim)
        else:
            weight_ = weight

        grad_input = F.linear(grad_output_, weight_.t())
        
        if bfp_conf.big:
            grad_input_ = make_groups_tensor(grad_input.clone().detach(), bfp_conf.big_bit,bfp_conf.big_dim)
        else:
            grad_input_ = grad_input

        if bfp_conf.bwo:
            # Regroup if bwo / bio grouping configuration is different!
            if (bfp_conf.bwo_bit != bfp_conf.bio_bit or bfp_conf.bwo_dim != bfp_conf.bio_dim):
                grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bwo_bit, bfp_conf.bwo_dim)
            else:
                grad_output_ = grad_output
        else: # If not grouping, use original type
            grad_output_ = grad_output
        ## Grouping input
        if bfp_conf.bwi:
            # Regroup if bwi / fi grouping configuration is different!
            if (bfp_conf.bwi_bit != bfp_conf.fi_bit or bfp_conf.bwi_dim != bfp_conf.fi_dim):
                input_ = make_groups_tensor(input.clone().detach(), bfp_conf.bwi_bit, bfp_conf.bwi_dim)
            else:
                input_ = input
        else:
            input_ = input
        

        # grad_weight = torch.empty((grad_output.shape[2], input.shape[2]),dtype=torch.float).cuda()
        if len(grad_output.shape == 2):
            grad_weight = gradient_linear_weight_2d(grad_output_.clone().detach(), input_.clone().detach(), weight.shape)
        elif len(grad_output.shape == 3):
            grad_weight = gradient_linear_weight_3d(grad_output_.clone().detach(), input_.clone().detach(), weight.shape)
        else:
            print("BFLinear - Backward ERROR: gradient weight dimention not supported")
            
        # Group the gradient of weight
        
        if bfp_conf.bwg:
            grad_weight = make_groups_tensor(grad_weight.clone().detach(), bfp_conf.bwg_bit, bfp_conf.bwg_dim)
        else:
            grad_weight = grad_weight
            
        if bfp_conf.bwg_boost != 1.0:
            grad_weight /= bfp_conf.bwg_boost

        if bias is not None:
            grad_bias = grad_output.sum(0)

        return grad_input_, grad_weight, grad_bias, None



# Blockfloat Linear
class BFPLinear(torch.nn.Module):
    def __init__(self,
                in_features: int,
                out_features: int,
                bfp_conf: BFPConf,
                bias=True):
        super(BFPLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bfp_conf = bfp_conf

        # Weight parameters, should be grouped with few numbers
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features).cuda())

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features).cuda())
        else:
            self.bias = None
            # self.register_paramter('bias', None)
        
        # Initialize weights manually
        self.weight.data.uniform_(-0.1, 0.1)
        if self.bias is not None:
            self.bias.data.uniform_(-0.1, 0.1)
    
    def forward(self, input):
        return BFPLinearFunction.apply(input, self.weight, self.bias, self.bfp_conf)
    
    def extra_repr(self):
        s = ('{in_features}, {out_features}')
        s += ', bfp_conf=({bfp_conf})'
        if self.bias is None:
            s += ', bias=False'
        else:
            s += ', bias=True'
        return s.format(**self.__dict__)


from torch.utils.cpp_extension import load
cudnn_convolution = load(name="cudnn_convolution", sources=["./extensions/cudnn_convolution.cpp"], build_directory= "./extensions/", verbose=True)

# Blockfloat Convolution Function
# TODO : Implement Conv2d Operation
# https://discuss.pytorch.org/t/implementing-a-custom-convolution-using-conv2d-input-and-conv2d-weight/18556/7
class BFPConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, bfp_conf=None, stride=1, padding=0, dilation=1, groups=1):
        # print("= Forward:",input.shape, weight.shape, stride, padding, dilation, groups)
        # Grouping input and weight
        if bfp_conf.fi:
            input = make_groups_tensor(input, bfp_conf.fi_bit, bfp_conf.fi_dim, 0)
        
        if bfp_conf.fw:
            weight_ = make_groups_tensor(weight.clone().detach(), bfp_conf.fw_bit, bfp_conf.fw_dim, 1)
        else:
            weight_ = weight

        ctx.stride = stride
        ctx.padding = padding
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.bfp_conf = bfp_conf

        ctx.save_for_backward(input, weight_, bias)

        # Compute Convolution
        output = F.conv2d(input, weight_, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        # Grouping Output
        if bfp_conf.fo:
            output = make_groups_tensor(output.clone().detach(), bfp_conf.fo_bit, bfp_conf.fo_dim, 2)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and configs
        input, weight, bias = ctx.saved_variables
        stride = ctx.stride
        padding = ctx.padding
        dilation = ctx.dilation
        groups = ctx.groups
        bfp_conf = ctx.bfp_conf

        # print("= Backward:",grad_output.shape, stride, padding, dilation, groups)
        grad_input = grad_weight = grad_bias = None

        # Calculate Input Gradient
        ## Grouping grad_output
        if bfp_conf.bio:
            grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bio_bit, bfp_conf.bio_dim, 10)
        else: # Apply original gradient if grad_output is not grouped
            grad_output_ = grad_output
        ## Grouping weight
        if bfp_conf.biw:
            if (bfp_conf.biw_bit != bfp_conf.fw_bit or bfp_conf.biw_dim != bfp_conf.fw_dim):
                weight_ = make_groups_tensor(weight.clone().detach(), bfp_conf.biw_bit, bfp_conf.biw_dim, 11)
            else:
                weight_ = weight
        else:
            weight_ = weight
        ## Do the convolution
        if ctx.needs_input_grad[0]: # First Layer's grad_input will be None
            # grad_input = torch.nn.grad.conv2d_input(input.shape, weight_, grad_output_, stride, padding, dilation, groups)
            grad_input = cudnn_convolution.convolution_backward_input(input.shape, weight_, grad_output_, stride, padding, dilation, groups, False, False, False)

        ## Grouping output grad_input
        if bfp_conf.big and grad_input != None:
            grad_input_ = make_groups_tensor(grad_input, bfp_conf.big_bit,bfp_conf.big_dim, 12)
        else:
            grad_input_ = grad_input

        # Calculate Weight Gradient (2D Convolution, Depthwise Convolution)
        ## Grouping grad_output
        
        if bfp_conf.bwo:
            # Regroup if bwo / bio grouping configuration is different!
            if (bfp_conf.bwo_bit != bfp_conf.bio_bit or bfp_conf.bwo_dim != bfp_conf.bio_dim):
                grad_output_ = make_groups_tensor(grad_output.clone().detach(), bfp_conf.bwo_bit, bfp_conf.bwo_dim, 20)
        else: # If not grouping, use original type
            grad_output_ = grad_output
        ## Grouping input - it's not grad_input, right?
        if bfp_conf.bwi:
            # Regroup if bwi / fi grouping configuration is different!
            if (bfp_conf.bwi_bit != bfp_conf.fi_bit or bfp_conf.bwi_dim != bfp_conf.fi_dim):
                input_ = make_groups_tensor(input, bfp_conf.bwi_bit, bfp_conf.bwi_dim, 21)
            else:
                input_ = input
        else:
            input_ = input
        
        ## Do the convolution
        if ctx.needs_input_grad[1]:
            # grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output_, stride, padding, dilation, groups)
            grad_weight = cudnn_convolution.convolution_backward_weight(input_, weight.shape, grad_output_, stride, padding, dilation, groups, False, False, False)
        # Group the gradient of weight
        if bfp_conf.bwg and grad_weight != None:
            grad_weight_ = make_groups_tensor(grad_weight, bfp_conf.bwg_bit, bfp_conf.bwg_dim, 22)
        else:
            grad_weight_ = grad_weight

        # Apply weaken gradient if weight gradient boost is applied
        # if bfp_conf.bwg_boost != 1.0:
        #     grad_weight /= bfp_conf.bwg_boost

        # TODO : Add Bias Grouping / or is it needed?
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output_.sum(dim=(0,2,3)).squeeze(0)
        
        return grad_input_, grad_weight_, grad_bias, None, None, None, None, None

# Blockfloat Convolution
class BFPConv2d(torch.nn.Module):
    def __init__(self,
                in_channels: int,
                out_channels: int,
                kernel_size: _size_2_t,
                bfp_conf: BFPConf, 
                stride: _size_2_t = 1,
                padding: _size_2_t = 0,
                dilation: _size_2_t = 1,
                groups: int = 1,
                bias: bool = True,
                padding_mode: str = 'zeros'):
        super(BFPConv2d, self).__init__()
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

        self.bfp_conf = bfp_conf
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
        return BFPConv2dFunction.apply(input, self.weight, self.bias, self.bfp_conf, self.stride, self.padding, self.dilation, self.groups)
    
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
        s += ', bfp_conf=({bfp_conf})'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)