import numpy as np

from models.attention import SE_Block
from models.networks_gated import *


class ResnetStackedArchitecture(nn.Module):

    def __init__(self, opt=None, input_nc=3):
        super(ResnetStackedArchitecture, self).__init__()
        
        # architecture parameters
        self.F           = 256 if not opt else opt.resnet_F
        self.B           = 16 if not opt else opt.resnet_B
        self.kernel_size = 3
        self.padding_size= 1
        self.scale_res   = 1
        self.dropout     = True
        if opt.no_64C:
            self.use_64C = False
        else:
            self.use_64C = True

        model = [nn.Conv2d(input_nc, self.F, kernel_size=self.kernel_size, padding=self.padding_size, bias=True),
                 nn.ReLU(True)]
        # generate a given number of blocks
        for i in range(self.B):
            model += [ResnetBlock(self.F, size=256, use_dropout=self.dropout, use_bias=True,
                                  res_scale=self.scale_res, use_attention=opt.use_attention)]

        # adding in intermediate mapping layer from self.F to 64 channels for STGAN pre-training
        if self.use_64C:
            model += [nn.Conv2d(self.F, 64, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
            model += [nn.ReLU(True)]
        if self.dropout:
            model += [nn.Dropout(0.2)]

        if self.use_64C:
            model += [nn.Conv2d(64, 1, kernel_size=self.kernel_size, padding=self.padding_size, bias=True)]
        
        self.model = nn.Sequential(*model)

    def forward(self, input):
        # long-skip connection: add cloudy MS input (excluding the trailing two SAR channels) and model output
        return self.model(input)   # + self.use_long*input[:, :(-2*self.use_SAR), ...]


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_dim, out_dim=None, use_dropout=False, use_bias=False, use_attention=False,
                 res_scale=1.0, late_relu=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.res_scale = res_scale
        if out_dim is None:
            out_dim = in_dim
        self.conv_block = self.build_conv_block(in_dim, out_dim, use_bias, use_dropout, use_attention)

        self.block_output_relu = None
        if late_relu:
            self.block_output_relu = nn.ReLU(True)

        self.skip_conv = None
        if out_dim != in_dim:
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size=1),
                nn.BatchNorm2d(out_dim),
            )

    def build_conv_block(self, in_dim, out_dim, use_bias, use_dropout, use_attention):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        conv_block += [nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
                       nn.BatchNorm2d(out_dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        conv_block += [nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1, bias=use_bias),
                       nn.BatchNorm2d(out_dim)]

        if use_attention:
            conv_block += [SE_Block(out_dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        if self.skip_conv is not None:
            x = self.skip_conv(x) + self.res_scale * self.conv_block(x)
        else:
            x = x + self.res_scale * self.conv_block(x)

        if self.block_output_relu is not None:
            x = self.block_output_relu(x)

        return x


class ResnetBlock3D(nn.Module):
    """Define a Resnet block"""

    def __init__(self, in_dim, out_dim=None, norm_layer='BatchNorm3D', use_dropout=False,
                 use_bias=True, use_attention=False, res_scale=1.0, late_relu=True):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock3D, self).__init__()
        self.res_scale = res_scale
        self.late_relu = late_relu
        if out_dim is None:
            out_dim = in_dim

        self.conv_block = self.build_conv_block(in_dim, out_dim, norm_layer, use_bias, use_dropout)

        self.skip_conv = None
        if out_dim != in_dim:
            self.skip_conv = nn.Sequential(
                nn.Conv3d(in_dim, out_dim, kernel_size=1, stride=1),
                nn.BatchNorm3d(out_dim),
            )

    def build_conv_block(self, in_dim, out_dim, norm_layer, use_bias, use_dropout):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, possibly normalisation layer, and a non-linearity layer (ReLU))
        """
        conv_block = []

        conv_block += [nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))]
        if norm_layer == 'BatchNorm3D':
            conv_block += [nn.Conv3d(in_dim, out_dim, kernel_size=3, bias=use_bias),
                           nn.BatchNorm3d(out_dim), nn.ReLU(True)]
        else:
            conv_block += [nn.Conv3d(in_dim, out_dim, kernel_size=3, bias=use_bias), nn.ReLU(True)]

        if use_dropout:
            conv_block += [nn.Dropout(0.2)]

        conv_block += [nn.ReplicationPad3d((1, 1, 1, 1, 1, 1))]
        if norm_layer == 'BatchNorm3D':
            conv_block += [nn.Conv3d(out_dim, out_dim, kernel_size=3, bias=use_bias), nn.BatchNorm3d(out_dim)]
        else:
            conv_block += [nn.Conv3d(out_dim, out_dim, kernel_size=3, bias=use_bias)]

        if self.late_relu:
            self.block_output_relu = nn.ReLU(True)
        else:
            conv_block += [nn.ReLU(True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        if self.skip_conv is not None:
            x = self.skip_conv(x) + self.res_scale * self.conv_block(x)
        else:
            x = x + self.res_scale * self.conv_block(x)
        if self.late_relu:
            x = self.block_output_relu(x)
        return x


def get_pad(in_, ksize, stride, atrous=1):
    out_ = np.ceil(float(in_) / stride)
    return int(np.ceil(((out_ - 1) * stride + atrous * (ksize - 1) + 1 - in_) / 2))
