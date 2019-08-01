from torch import nn
from torch.autograd import grad
import torch
DIM=64
OUTPUT_DIM=64*64*3
MAX_DIM = 1024
class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.he_init = he_init
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)

    def forward(self, input):
        output = self.conv(input)
        return output

class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(ConvMeanPool, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output

class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True):
        super(MeanPoolConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output

class DepthToSpace(nn.Module):
    def __init__(self, block_size):
        super(DepthToSpace, self).__init__()
        self.block_size = block_size
        self.block_size_sq = block_size*block_size

    def forward(self, input):
        output = input.permute(0, 2, 3, 1)
        (batch_size, input_height, input_width, input_depth) = output.size()
        output_depth = int(input_depth / self.block_size_sq)
        output_width = int(input_width * self.block_size)
        output_height = int(input_height * self.block_size)
        t_1 = output.reshape(batch_size, input_height, input_width, self.block_size_sq, output_depth)
        spl = t_1.split(self.block_size, 3)
        stacks = [t_t.reshape(batch_size,input_height,output_width,output_depth) for t_t in spl]
        output = torch.stack(stacks,0).transpose(0,1).permute(0,2,1,3,4).reshape(batch_size,output_height,output_width,output_depth)
        output = output.permute(0, 3, 1, 2)
        return output


class UpSampleConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, he_init = True, bias=True):
        super(UpSampleConv, self).__init__()
        self.he_init = he_init
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size, he_init = self.he_init, bias=bias)
        self.depth_to_space = DepthToSpace(2)

    def forward(self, input):
        output = input
        output = torch.cat((output, output, output, output), 1)
        output = self.depth_to_space(output)
        output = self.conv(output)
        return output


class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, resample=None, hw=DIM):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.bn1 = None
        self.bn2 = None
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'up':
            self.bn1 = nn.BatchNorm2d(input_dim)
            self.bn2 = nn.BatchNorm2d(output_dim)
        elif resample == None:
            #TODO: ????
            self.bn1 = nn.BatchNorm2d(output_dim)
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        else:
            raise Exception('invalid resample value')

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'up':
            self.conv_shortcut = UpSampleConv(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = UpSampleConv(input_dim, output_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(output_dim, output_dim, kernel_size = kernel_size)
        elif resample == None:
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1, he_init = False)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
        else:
            raise Exception('invalid resample value')

    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
            shortcut = input
        else:
            shortcut = self.conv_shortcut(input)

        output = input
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.conv_1(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv_2(output)

        return shortcut + output

class ReLULayer(nn.Module):
    def __init__(self, n_in, n_out):
        super(ReLULayer, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.linear = nn.Linear(n_in, n_out)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.linear(input)
        output = self.relu(output)
        return output

class FCGenerator(nn.Module):
    def __init__(self, FC_DIM=512):
        super(FCGenerator, self).__init__()
        self.relulayer1 = ReLULayer(128, FC_DIM)
        self.relulayer2 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer3 = ReLULayer(FC_DIM, FC_DIM)
        self.relulayer4 = ReLULayer(FC_DIM, FC_DIM)
        self.linear = nn.Linear(FC_DIM, OUTPUT_DIM)
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.relulayer1(input)
        output = self.relulayer2(output)
        output = self.relulayer3(output)
        output = self.relulayer4(output)
        output = self.linear(output)
        output = self.tanh(output)
        return output

class GoodGenerator(nn.Module):
    def __init__(self, dim=DIM,output_dim=OUTPUT_DIM):
        super(GoodGenerator, self).__init__()

        self.dim = dim
        if output_dim == 64:
            self.in_order = [8,8,4,2]
            self.out_order = [8,4,2,1]
        elif output_dim == 128:
            self.in_order = [8,8,4,4,2]
            self.out_order = [8,4,4,2,1]
        elif output_dim == 256:
            self.in_order = [8,8,4,4,2,2]
            self.out_order = [8,4,4,2,2,1]
        else:
            raise ValueError("Unsupported resolution: {}".format(output_dim))

        self.ln = nn.Linear(128, 4*4*self.in_order[0]*self.dim)
        self.res_block = []
        for index in range(len(self.in_order)):
            nf_in = min(MAX_DIM,self.in_order[index] * self.dim)
            nf_out = min(MAX_DIM,self.out_order[index] * self.dim)
            self.res_block.append(
                ResidualBlock(nf_in,nf_out, 3, resample='up')
            )
        self.res_block = nn.Sequential(*self.res_block)
        self.bn  = nn.BatchNorm2d(self.dim)

        self.conv1 = MyConvo2d(1*self.dim, 3, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, input):
        output = self.ln(input.view(-1,128))
        output = output.view(-1, self.in_order[0]*self.dim, 4, 4)
        output = self.res_block(output)

        output = self.bn(output)
        output = self.relu(output)
        output = self.conv1(output)
        output = self.tanh(output)
        # output = output.view(-1, OUTPUT_DIM)
        return output

class GoodDiscriminator(nn.Module):
    def __init__(self, dim=DIM,input_dim = 64):
        super(GoodDiscriminator, self).__init__()

        self.dim = dim
        self.input_dim = input_dim
        if input_dim == 64:
            self.in_order = [1,2,4,8]
            self.out_order = [2,4,8,8]
        elif input_dim == 128:
            self.in_order = [1,2,4,4,8]
            self.out_order = [2,4,4,8,8]
        elif input_dim == 256:
            self.in_order = [1,2,2,4,4,8]
            self.out_order = [2,2,4,4,8,8]
        else:
            raise ValueError("Unsupported resolution: {}".format(input_dim))

        self.conv1 = MyConvo2d(3, self.dim, 3, he_init = False)
        self.res_block = []
        for index in range(len(self.in_order)):
            nf_in = min(MAX_DIM,self.in_order[index] * self.dim)
            nf_out = min(MAX_DIM,self.out_order[index] * self.dim)

            self.res_block.append(
                ResidualBlock(nf_in, nf_out, 3, resample='down', hw=input_dim//(2**(index)))
            )
        self.res_block = nn.Sequential(*self.res_block)

        self.ln1 = nn.Linear(4*4*self.out_order[-1]*self.dim, 1)

    def forward(self, input):
        output = input.contiguous()
        # output = output.view(-1, 3, DIM, DIM)
        output = self.conv1(output)
        output = self.res_block(output)
        output = output.view(-1, 4*4*self.out_order[-1]*self.dim)
        output = self.ln1(output)
        output = output.view(-1)
        return output


class GoodDiscriminator_relation(nn.Module):
    def __init__(self, dim=DIM, input_dim=64):
        super(GoodDiscriminator_relation, self).__init__()

        self.dim = dim
        self.input_dim = input_dim
        if input_dim == 64:
            self.in_order = [1,2,4,8]
            self.out_order = [2,4,8,8]
        elif input_dim == 128:
            self.in_order = [1,2,4,4,8]
            self.out_order = [2,4,4,8,8]
        elif input_dim == 256:
            self.in_order = [1,2,2,4,4,8]
            self.out_order = [2,2,4,4,8,8]
        else:
            raise ValueError("Unsupported resolution: {}".format(input_dim))

        self.conv1 = MyConvo2d(3, self.dim, 3, he_init=False)
        self.res_block = []
        for index in range(len(self.in_order)):
            nf_in = min(MAX_DIM,self.in_order[index] * self.dim)
            nf_out = min(MAX_DIM,self.out_order[index] * self.dim)

            self.res_block.append(
                ResidualBlock(nf_in, nf_out, 3, resample='down',
                              hw=input_dim // (2 ** (index)))
            )
        self.res_block = nn.Sequential(*self.res_block)

        self.relation = nn.Sequential(
        #    nn.BatchNorm2d(self.out_order[-1] * self.dim),
            nn.Conv2d(self.out_order[-1] * self.dim,self.out_order[-1] * self.dim,kernel_size=3,stride=1,padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(self.out_order[-1] * self.dim,self.out_order[-1] * self.dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(self.out_order[-1] * self.dim,self.out_order[-1] * self.dim, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.1,inplace=True),
            nn.Conv2d(self.out_order[-1] * self.dim,self.out_order[-1] * self.dim, kernel_size=3, stride=1, padding=1)
        #    nn.Conv2d(self.out_order[-1] * self.dim//2,self.out_order[-1] * self.dim//2,kernel_size=3,stride=1,padding=1),
        )

        self.ln1 = nn.Linear(4 * 4 * self.out_order[-1] * self.dim, 1)



    def forward(self, input1,input2):
        input1 = input1.contiguous()
        # output = output.view(-1, 3, DIM, DIM)
        input1 = self.conv1(input1)
        input1 = self.res_block(input1)
        
        input2 = input2.contiguous()
        # output = output.view(-1, 3, DIM, DIM)
        input2 = self.conv1(input2)
        input2 = self.res_block(input2)
        
        input1 = self.relation(input1+input2)

        input1 = input1.view(-1, 4 * 4 * self.out_order[-1] * self.dim)


        output = self.ln1(input1)
        output = output.view(-1)
        return output