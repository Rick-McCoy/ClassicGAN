import torch
import queue
import numpy as np
from torch.utils.checkpoint import checkpoint

class CausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, padding=1, 
                                    dilation=1, bias=False)
    
    def forward(self, x):
        return self.conv(x)[:, :, :-1]

class DilatedCausalConv1d(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, dilation=dilation, 
                                    bias=False)
        self.sample_conv = torch.nn.Conv1d(in_channels, out_channels, 
                                    kernel_size=2, bias=False)
        self.sample_conv.weight = self.conv.weight

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(torch.nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, dilation):
        super(ResidualBlock, self).__init__()
        self.dilation = dilation
        self.dilated_conv = DilatedCausalConv1d(residual_channels, residual_channels, dilation=dilation)
        self.tanh_conv = torch.nn.Conv1d(residual_channels, dilation_channels, 1)
        self.sigmoid_conv = torch.nn.Conv1d(residual_channels, dilation_channels, 1)
        self.residual_conv = torch.nn.Conv1d(dilation_channels, residual_channels, 1)
        self.skip_conv = torch.nn.Conv1d(dilation_channels, skip_channels, 1)
        self.gate_tanh = torch.nn.Tanh()
        self.gate_sigmoid = torch.nn.Sigmoid()
        self.queue = queue.Queue(dilation)

    def forward(self, x, skip_size, sample=False):
        if sample:
            output = self.dilated_conv.sample_conv(x)
        else:
            output = self.dilated_conv(x)

        gated_tanh = self.tanh_conv(output)
        gated_sigmoid = self.sigmoid_conv(output)
        gated_tanh = self.gate_tanh(gated_tanh)
        gated_sigmoid = self.gate_sigmoid(gated_sigmoid)
        gated = gated_tanh * gated_sigmoid

        output = self.residual_conv(gated)
        output += x[:, :, -output.size()[2]:]

        skip = self.skip_conv(gated)
        skip = skip[:, :, -skip_size:]
        return output, skip

class ResidualStack(torch.nn.Module):
    def __init__(
            self, 
            layer_size, 
            stack_size, 
            residual_channels, 
            dilation_channels, 
            skip_channels
        ):
        super(ResidualStack, self).__init__()
        self.layer_size = layer_size
        self.stack_size = stack_size
        self.res_blocks = torch.nn.ModuleList(
            self.stack_res_blocks(
                residual_channels, 
                dilation_channels, 
                skip_channels
            )
        )
        
    def stack_res_blocks(self, residual_channels, dilation_channels, skip_channels):
        dilations = [2 ** i for i in range(self.layer_size)] * self.stack_size
        res_blocks = [ResidualBlock(residual_channels, dilation_channels, skip_channels, dilation) for dilation in dilations]
        return res_blocks
    
    def forward(self, x, skip_size):
        output = x
        res_sum = 0
        for res_block in self.res_blocks:
            output, skip = res_block(output, skip_size)
            res_sum += skip
        return res_sum

    def sample_forward(self, x):
        output = x
        res_sum = 0
        for res_block in self.res_blocks:
            top = res_block.queue.get()
            res_block.queue.put(output)
            full = torch.cat((top, output), dim=2) # pylint: disable=E1101
            output, skip = res_block(full, 1, sample=True)
            res_sum += skip
        return res_sum

    def fill_queues(self, x):
        for res_block in self.res_blocks:
            with res_block.queue.mutex:
                res_block.queue.queue.clear()
            for i in range(-res_block.dilation - 1, -1):
                    res_block.queue.put(x[:, :, i:i + 1])
            x, _ = res_block(x, 1)

class PostProcess(torch.nn.Module):
    def __init__(self, skip_channels, end_channels, out_channels):
        super(PostProcess, self).__init__()
        self.conv1 = torch.nn.Conv1d(skip_channels, end_channels, 1)
        self.conv2 = torch.nn.Conv1d(end_channels, out_channels, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        output = self.relu(x)
        output = self.conv1(output)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.sigmoid(output)
        return output

class Wavenet(torch.nn.Module):
    def __init__(
            self, 
            layer_size, 
            stack_size, 
            channels, 
            residual_channels, 
            dilation_channels, 
            skip_channels, 
            end_channels, 
            out_channels
        ):
        super(Wavenet, self).__init__()
        self.receptive_field = self.calc_receptive_field(layer_size, stack_size)
        self.causal = CausalConv1d(channels, residual_channels)
        self.res_stacks = ResidualStack(
            layer_size, 
            stack_size, 
            residual_channels, 
            dilation_channels, 
            skip_channels
        )
        self.post = PostProcess(skip_channels, end_channels, out_channels)
    
    @staticmethod
    def calc_receptive_field(layer_size, stack_size):
        layers = [2 ** i for i in range(layer_size)] * stack_size
        return np.sum(layers)

    def calc_output_size(self, x):
        output_size = x.size()[2] - self.receptive_field
        return output_size

    def forward(self, x):
        output_size = self.calc_output_size(x)
        output = self.causal(x)
        output = self.res_stacks(output, output_size)
        output = self.post(output)
        return output
    
    def sample_forward(self, x):
        output = self.causal(x)[:, :, 1:]
        output = self.res_stacks.sample_forward(output)
        output = self.post(output)
        return output

    def fill_queues(self, x):
        x = self.causal(x)
        self.res_stacks.fill_queues(x)
