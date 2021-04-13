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

from blockfunc import make_groups_tensor, set_mantissa_tensor
from functions import BFConf

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
    def forward(ctx, input, weight, bf_conf, bias):    
        # Grouping input and weight
        if DEFAULT_CUDA:
            input = make_groups_tensor(input.cpu(), bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir).cuda()
            weight = make_groups_tensor(weight.cpu(), bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir).cuda()
        else:
            input = make_groups_tensor(input, bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir)
            weight = make_groups_tensor(weight, bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir)

        # Save context to use on backward
        bf_confs = torch.from_numpy(np.array([
            bf_conf.g_o_bit, bf_conf.g_o_sz,
            bf_conf.g_o_dir[0], bf_conf.g_o_dir[1],
            bf_conf.g_i_bit, bf_conf.g_w_bit, bf_conf.g_b_bit]))
        ctx.save_for_backward(input, weight, bias, bf_confs)
        
        # Compute FC and return
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)

        # Set mantissa because result of computation is preseted bits
        if DEFAULT_CUDA:
            output = set_mantissa_tensor(output.cpu(), bf_conf.f_o_bit).cuda()
        else:
            output = set_mantissa_tensor(output, bf_conf.f_o_bit)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors
        # input, weight, bias, confs = ctx.saved_tensors
        input, weight, bias, bf_confs = ctx.saved_tensors
        bf_confs = bf_confs.numpy()
        g_o_dir = [0,0]
        g_o_bit, g_o_sz, g_o_dir[0], g_o_dir[1], g_i_bit, g_w_bit, g_b_bit = bf_confs
        g_o_dir = tuple(g_o_dir)

        # Output Gradient Grouping
        if DEFAULT_CUDA:
            grad_output = make_groups_tensor(grad_output.cpu(), g_o_bit, g_o_sz, g_o_dir).cuda()
        else:
            grad_output = make_groups_tensor(grad_output, 8, g_o_bit, g_o_sz, g_o_dir)

        # Calculate gradients
        grad_input = grad_weight = grad_bias = None
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input)

        # Set mantissa because result of computation is preseted bits
        if DEFAULT_CUDA:
            grad_input = set_mantissa_tensor(grad_input.cpu(), g_i_bit).cuda()
            grad_weight = set_mantissa_tensor(grad_weight.cpu(), g_w_bit).cuda()
        else:
            grad_input = set_mantissa_tensor(grad_input, g_i_bit)
            grad_weight = set_mantissa_tensor(grad_weight, g_w_bit)
        

        if bias is not None:
            grad_bias = grad_output.sum(0)
            if DEFAULT_CUDA:
                grad_bias = set_mantissa_tensor(grad_bias.cpu(), g_b_bit).cuda()
            else:
                grad_bias = set_mantissa_tensor(grad_bias, g_b_bit)

        return grad_input, grad_weight, None, grad_bias, None

# Blockfloat Linear
class BFLinear(torch.nn.Module):
    def __init__(self,
                input_features: int,
                output_features: int,
                bf_conf: BFConf):
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
        return BFLinearFunction.apply(input, self.weight, self.bf_conf, self.bias)
    
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
        if bf_conf.f_i:
            input = make_groups_tensor(input, bf_conf.f_i_bit, bf_conf.f_i_sz, bf_conf.f_i_dir)
        if bf_conf.f_w:
            weight = make_groups_tensor(weight, bf_conf.f_w_bit, bf_conf.f_w_sz, bf_conf.f_w_dir)

        # Save arguments to context to use on backward
        # WARNING : if stride, padding, dilation etc is array, this will not work properly!!!!
        confs = torch.from_numpy(np.array([stride, padding, dilation, groups]))
        b_w = 1 if bf_conf.b_w else 0
        b_og = 1 if bf_conf.b_w else 0
        b_ig = 1 if bf_conf.b_w else 0
        b_wg = 1 if bf_conf.b_w else 0
        
        bf_confs = torch.from_numpy(np.array([
            b_w, bf_conf.b_w_bit, bf_conf.b_w_sz, bf_conf.b_w_dir,
            b_og, bf_conf.b_og_bit, bf_conf.b_og_sz, bf_conf.b_og_dir,
            b_ig, bf_conf.b_ig_bit, bf_conf.b_ig_sz, bf_conf.b_ig_dir,
            b_wg, bf_conf.b_wg_bit, bf_conf.b_wg_sz, bf_conf.b_wg_dir]))
        ctx.save_for_backward(input, weight, bias, confs, bf_confs)

        # Compute Convolution
        output = F.conv2d(input, weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        # Grouping output
        if bf_conf.f_o:
            output = make_groups_tensor(output, bf_conf.f_o_bit, bf_conf.f_o_sz, bf_conf.f_o_dir)

        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Load saved tensors and configs
        input, weight, bias, confs, bf_confs = ctx.saved_variables
        confs, bf_confs = confs.numpy(), bf_confs.numpy()
        stride, padding, dilation, groups = confs
        b_w, b_w_bit, b_w_sz, b_w_dir, b_og, b_og_bit, b_og_sz, b_og_dir, b_ig, b_ig_bit, b_ig_sz, b_ig_dir, b_wg, b_wg_bit, b_wg_sz, b_wg_dir = bf_confs
        b_og = True if b_og > 0 else False
        b_ig = True if b_ig > 0 else False
        b_wg = True if b_wg > 0 else False
        b_w = True if b_w > 0 else False
        # print("= Backward:",grad_output.shape, stride, padding, dilation, groups)
        
        # output gradient grouping
        if b_og:
            grad_output = make_groups_tensor(grad_output, b_og_bit, b_og_sz, b_og_dir)
        if b_w:
            weight = make_groups_tensor(weight, b_w_bit, b_w_sz, b_w_dir)
        
        # Calculate Gradient
        grad_input = grad_weight = grad_bias = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding, dilation, groups)           
        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding, dilation, groups)

        # Grouping input and weight
        if b_ig:
            grad_input = make_groups_tensor(grad_input, b_ig_bit, b_ig_sz, b_ig_dir)
        if b_wg:
            grad_weight = make_groups_tensor(grad_weight, b_wg_bit, b_wg_sz, b_wg_dir)
        
        # WARNING : Bias maybe buggy, remove if it is buggy
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(dim=(0,2,3)).squeeze(0)
            # TODO : Bias Grouping
        
        return grad_input, grad_weight, grad_bias, None, None, None, None, None

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
        self.kernel_size = kernel_size
        self.bf_conf = bf_conf
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
