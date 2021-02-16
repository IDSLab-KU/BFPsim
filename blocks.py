import torch
import torch.nn as nn

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
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        input = torch.zeros(input.size())
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

# Blockfloat Linear
class BFLinear(torch.nn.Module):
    def __init__(self, input_features, output_features, bias=True):
        super(BFLinear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features

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
class BFConv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
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


# Blockfloat Convolution
class BFConv2d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, \
        padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(BFConv2d, self).__init__()
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.group = group
        # From torch/nn/modules/conv.py
        self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels // groups, *kernel_size))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
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
        return BFConv2dFunction.apply(input, self.weight, self.bias)

    def extra_repr(self):
        # From /torch/nn/modules/conv.py
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)
