import torch
import math
import torch.nn as nn

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.lw = LinearWise(321*481*3,bias=False)
    
    def forward(self, x):
        x = x.view(x.size(0),-1)
        x = self.lw(x)
        x = x.view(1, 3, 321, 481)
        return x


class LinearWise(nn.Module):
    def __init__(self, in_features, bias=True):
        super(LinearWise, self).__init__()
        self.in_features = in_features

        self.weight = nn.Parameter(torch.Tensor(self.in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(self.in_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1
        self.weight.data.uniform_(stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(stdv, stdv)

    def forward(self, input):
        x = input * self.weight
        if self.bias is not None:
            x = x + self.bias
        return x

if __name__=='__main__':
    model = net()
    print(model)
    input = torch.randn(1, 3, 321, 481)
    out = model(input)
    print(out.size())
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
    p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')