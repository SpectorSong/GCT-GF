import functools
import tifffile as tif
import os
from torch.optim import lr_scheduler
from torch.nn import init

from models.networks_resnet import *
from models.networks_gated import *
from models.attention import ContextualAttention
from models.networks_restormer import TransformerBlock


###############################################################################
# Helper Functions
###############################################################################
def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)

    return norm_layer


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.5)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.5, patience=3)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, net_name, init_type='normal', init_gain=0.02):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize [%s] with %s' % (net_name, init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, net_name, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, net_name, init_type, init_gain=init_gain)
    return net


def define_G(device, gpu_ids=[], opt=None):
    """Create a generator
    """

    if opt.s1_channels in ['vv', 'vh']:
        input_nc = 2
    else:
        input_nc = 3

    if opt.netG == 'wpgct':
        net = WPGCT(opt=opt)
    elif opt.netG == 'wpgct_nodw':
        net = WPGCT_noDW(opt=opt)
    elif opt.netG == 'wpgct_nos1':
        net = WPGCT_noS1(opt=opt)
    elif opt.netG == 'stgan':
        net = ResnetGenerator(input_nc, opt.ngf, n_blocks=opt.n_blocks, use_dropout=not opt.no_dropout)
    elif opt.netG == 'dsen2cr':
        net = Resnet(device, n_blocks=opt.n_blocks, opt=opt)
    elif opt.netG == 'sen12mscrts':
        net = ResnetGenerator3DWithoutBottleneck(device, n_blocks=opt.n_blocks, opt=opt)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % opt.netG)

    # initiate the weights of STGAN
    model = init_net(net, 'Generator', opt.init_type, opt.init_gain, gpu_ids)

    return model


def define_D(input_nc, opt=None, gpu_ids=[]):
    """Create a discriminator

    Parameters:
        input_nc (int)     -- the number of channels in input images
        ndf (int)          -- the number of filters in the first conv layer
        netD (str)         -- the architecture's name: basic | n_layers | pixel
        n_layers_D (int)   -- the number of conv layers in the discriminator; effective when netD=='n_layers'
        norm (str)         -- the type of normalization layers used in the network.
        init_type (str)    -- the name of the initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Returns a discriminator

    Our current implementation provides three types of discriminators:
        [basic]: 'PatchGAN' classifier described in the original pix2pix paper.
        It can classify whether 70×70 overlapping patches are real or fake.
        Such a patch-level discriminator architecture has fewer parameters
        than a full-image discriminator and can work on arbitrarily-sized images
        in a fully convolutional fashion.

        [n_layers]: With this mode, you cna specify the number of conv layers in the discriminator
        with the parameter <n_layers_D> (default=3 as used in [basic] (PatchGAN).)

        [pixel]: 1x1 PixelGAN discriminator can classify whether a pixel is real or not.
        It encourages greater color diversity but has no effect on spatial statistics.

        [sn]: 'SN-PatchGAN' in paper: 'Free-Form Image Inpainting with Gated Convolution'.

    The discriminator has been initialized by <init_net>. It uses Leakly RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=opt.norm)

    if opt.netD == 'basic':  # default PatchGAN classifier
        net = NLayerDiscriminator(input_nc, opt.ndf, n_layers=3, norm_layer=norm_layer)
    elif opt.netD == 'n_layers':  # more options
        net = NLayerDiscriminator(input_nc, opt.ndf, opt.n_layers_D, norm_layer=norm_layer)
    elif opt.netD == 'pixel':  # classify if each pixel is real or fake
        net = PixelDiscriminator(input_nc, opt.ndf, norm_layer=norm_layer)
    elif opt.netD == 'sn':
        net = SNDiscriminator(input_nc, opt.ndf)
    elif opt.netD == 'mbsn':
        net = MultiBranchSNDiscriminator(input_nc, opt.ndf, opt.n_input_samples)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, 'Discriminator', opt.init_type, opt.init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real, target=None, is_G=False, gp=None):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - typically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            raise Exception('GANLoss type is not supported')
        return loss


class ResnetGenerator(nn.Module):
    """stgan
    """

    def __init__(self, input_nc, ngf=64, n_blocks=6, use_bias=True, use_dropout=False):
        """Construct a Resnet-based generator
        Parameters:
            input_nc (int)      -- the number of channels in input images
            ngf (int)           -- the number of filters in the last conv layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator, self).__init__()

        model_initial = [nn.Conv2d(input_nc, ngf, kernel_size=7, padding=get_pad(256, 7, 1), bias=use_bias),
                         nn.BatchNorm2d(ngf),
                         nn.ReLU(True)]

        n_downsampling = 2
        model_intermediate = []
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model_intermediate += [
                nn.Conv2d(2 * ngf * mult, 2 * ngf * mult * 2, 3, 2, padding=get_pad(256//mult, 3, 2), bias=use_bias),
                nn.BatchNorm2d(2 * ngf * mult * 2),
                nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):  # add ResNet blocks
            model_intermediate += [ResnetBlock(2 * ngf * mult, use_dropout=use_dropout, use_bias=use_bias)]

        model_final = []
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model_final += [nn.ConvTranspose2d(6 * ngf * mult, int(6 * ngf * mult / 2), kernel_size=3, stride=2,
                                               padding=get_pad(64*(2**i), 3, 1), output_padding=1, bias=use_bias),
                            nn.BatchNorm2d(int(6 * ngf * mult / 2)),
                            nn.ReLU(True)]

        model_final += [nn.Conv2d(6 * ngf, 1, 7, 1, padding=get_pad(256, 7, 1), bias=use_bias)]
        model_final += [nn.Tanh()]

        self.model_initial = nn.Sequential(*model_initial)
        self.model_intermediate = nn.Sequential(*model_intermediate)
        self.model_final = nn.Sequential(*model_final)

    def forward(self, input):
        """Standard forward"""
        input_0 = input[0]
        input_1 = input[1]
        input_2 = input[2]
        output_0 = self.model_initial(input_0)
        output_1 = self.model_initial(input_1)
        output_2 = self.model_initial(input_2)

        intermediate_input_01 = torch.cat((output_0, output_1), 1)
        intermediate_input_02 = torch.cat((output_0, output_2), 1)
        intermediate_input_12 = torch.cat((output_1, output_2), 1)
        output_intermediate_0 = self.model_intermediate(intermediate_input_01)
        output_intermediate_1 = self.model_intermediate(intermediate_input_02)
        output_intermediate_2 = self.model_intermediate(intermediate_input_12)

        x = torch.cat((output_intermediate_0, output_intermediate_1, output_intermediate_2), 1)
        x_final = self.model_final(x)

        return x_final


class Resnet(nn.Module):
    """DSen2-CR
    """

    def __init__(self, device, n_blocks=6, opt=None):
        assert (n_blocks >= 0)
        super(Resnet, self).__init__()

        assert opt.n_input_samples == 1

        # get initial model, see networks_resnet.py
        # 16 blocks of 256 features, then mapping from 256 to 64 channels, and finally from 64 to 13 channels
        # (note: the last two CONV layers may or may not be taken according to line:
        # model_initial = nn.Sequential(*[m.model[i] for i in range(len(m.model) - 2)]))
        model_initial = ResnetStackedArchitecture(opt=opt)
        model_initial.to(device)
        self.initial_freezed = True
        self.n_channels = 256 if not opt else opt.resnet_F
        self.model_initial = model_initial

    def forward(self, input):
        """Standard forward"""

        """
        initial_output = []
        for each in input:
            initial_output.append(self.model_initial(each))
        x = torch.stack(initial_output, dim=2)
        output = self.model_final(x)
        """
        output = self.model_initial(input[0])
        return output


def factorial(num):
    if num == 0:
        return 1
    elif num > 0:
        f = 1
        for i in range(1, num + 1):
            f *= i
        return f
    else:
        raise Exception("Factorial must be larger than 0")


class ResnetGenerator3DWithoutBottleneck(nn.Module):
    """SEN12MS-CR-TS
    """

    def __init__(self, device, n_blocks=6, padding_type='zero', opt=None):
        """Construct a Resnet-based generator

        Parameters:
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(ResnetGenerator3DWithoutBottleneck, self).__init__()

        assert opt.n_input_samples >= 3

        self.opt = opt

        # opt.no_64C = True
        model_initial = ResnetStackedArchitecture(opt=opt)
        model_initial.to(device)
        self.n_channels = 256 if not opt else opt.resnet_F

        resnet_blocks = 5
        if opt.no_64C:
            final_channels = self.n_channels
        else:
            final_channels = 1
        model_final = [nn.Conv3d(final_channels, self.n_channels, kernel_size=(3, 3, 3), padding=1, bias=True),
                       nn.ReLU(True)]
        for i in range(resnet_blocks):
            model_final += [ResnetBlock3D(self.n_channels, norm_layer='none', use_bias=True,
                                          res_scale=0.1, use_attention=opt.use_attention)]

        # model_final += [ReflectionPad3D(0, 1)]
        model_final += [nn.Conv3d(self.n_channels, 1, kernel_size=(opt.n_input_samples, 3, 3), padding=(0, 1, 1))]
        model_final += [nn.Tanh()]

        self.model_initial = model_initial
        self.model_final = nn.Sequential(*model_final)

    def forward(self, input, t_weight=None):
        """Standard forward"""
        if t_weight:
            input_0 = input[0] * t_weight[0]
            input_1 = input[1] * t_weight[1]
            input_2 = input[2] * t_weight[2]
        else:
            input_0 = input[0]
            input_1 = input[1]
            input_2 = input[2]
        output_0 = self.model_initial(input_0)
        output_1 = self.model_initial(input_1)
        output_2 = self.model_initial(input_2)

        x = torch.stack([output_0, output_1, output_2], dim=2)
        final = self.model_final(x)

        if self.opt.use_branch_loss:
            return output_0, output_1, output_2, final.squeeze(2)
        else:
            return final.squeeze(2)


class WPGCT(nn.Module):

    def __init__(self, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(WPGCT, self).__init__()

        # assert opt.n_input_samples >= 3

        self.n_samples = opt.n_input_samples
        self.opt = opt

        if not opt.no_stage1:
            # --------------------------- coarse net ---------------------------
            self.model_initial = nn.Sequential(
                GatedConv2dWithActivation(4, opt.ngf, 5, 1, padding=get_pad(256, 5, 1)),
                # downsample 128
                GatedConv2dWithActivation(opt.ngf, 2 * opt.ngf, 3, 2, padding=get_pad(256, 3, 2)),
                GatedConv2dWithActivation(2 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
                # downsample to 64
                GatedConv2dWithActivation(2 * opt.ngf, 4 * opt.ngf, 3, 2, padding=get_pad(128, 3, 2)),
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
                # atrous convolution
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),

                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
                GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
                # upsample to 128
                GatedDeConv2dWithActivation(2, 4 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
                GatedConv2dWithActivation(2 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
                # upsample to 256
                GatedDeConv2dWithActivation(2, 2 * opt.ngf, opt.ngf, 3, 1, padding=get_pad(256, 3, 1)),
                GatedConv2dWithActivation(opt.ngf, opt.ngf, 3, 1, padding=get_pad(256, 3, 1)),
            )

            self.model_coarse_1 = nn.Sequential(
                nn.Conv3d(opt.ngf, opt.ngf, kernel_size=(opt.n_input_samples, 3, 3), padding=(0, 1, 1)),
                nn.BatchNorm3d(opt.ngf),
                nn.ReLU(True))
            self.model_coarse_2 = nn.Sequential(
                nn.Conv2d(opt.ngf, 1, kernel_size=3, padding=1),
                nn.Sigmoid()
            )

        if not opt.no_stage2:
            # --------------------------- align and refine net ---------------------------
            self.dw_start = nn.Conv2d(1, opt.embed_dim, 3, 1, 1)
            self.dw_down1 = nn.Sequential(
                # downsample to 128
                nn.Conv2d(opt.embed_dim, opt.embed_dim // 2, 3, 1, 1),
                nn.PixelUnshuffle(2)
            )
            self.dw_down2 = nn.Sequential(
                # downsample to 64
                nn.Conv2d(opt.embed_dim * 2, opt.embed_dim, 3, 1, 1),
                nn.PixelUnshuffle(2)
            )
            self.dw_down3 = nn.Sequential(
                # downsample to 32
                nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 2, 3, 1, 1),
                nn.PixelUnshuffle(2)
            )

            self.sar_start = nn.Sequential(
                nn.Conv2d(2, opt.embed_dim, 3, 1, 1),
                TransformerBlock(dim=opt.embed_dim, num_heads=8),
                # downsample to 128
                nn.Conv2d(opt.embed_dim, opt.embed_dim // 2, 3, 1, 1),
                nn.PixelUnshuffle(2),
                TransformerBlock(dim=opt.embed_dim * 2, num_heads=8),
                # downsample to 64
                nn.Conv2d(opt.embed_dim * 2, opt.embed_dim, 3, 1, 1),
                nn.PixelUnshuffle(2),
                TransformerBlock(dim=opt.embed_dim * 4, num_heads=8),
                # downsample to 32
                nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 2, 3, 1, 1),
                nn.PixelUnshuffle(2),
                TransformerBlock(dim=opt.embed_dim * 8, num_heads=8),
            )
            self.sar_up1 = nn.Sequential(
                TransformerBlock(dim=opt.embed_dim * 8, num_heads=8),
                # upsample to 64
                nn.Conv2d(opt.embed_dim * 8, opt.embed_dim * 16, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.sar_up2 = nn.Sequential(
                TransformerBlock(dim=opt.embed_dim * 4, num_heads=8),
                # upsample to 128
                nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 8, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.sar_up3 = nn.Sequential(
                TransformerBlock(dim=opt.embed_dim * 2, num_heads=8),
                # upsample to 256
                nn.Conv2d(opt.embed_dim * 2, opt.embed_dim * 4, 3, 1, 1),
                nn.PixelShuffle(2)
            )
            self.sar_final = TransformerBlock(dim=opt.embed_dim, num_heads=8)

            self.final_conv = nn.Conv2d(opt.embed_dim, 1, 3, 1, 1)
            self.out_activate = nn.Sigmoid()

    def forward(self, input, **kwargs):
        """Standard forward"""
        mask = None
        target_S1 = None
        if 'mask' in kwargs.keys():
            mask = kwargs['mask']
        if 'B_S1' in kwargs.keys():
            target_S1 = kwargs['B_S1']

        if not self.opt.no_stage1:
            initial_list = []
            for t in range(self.n_samples):
                output = self.model_initial(torch.cat([input[t], mask[t]], dim=1))
                initial_list.append(output)

            x = torch.stack(initial_list, dim=2)
            del initial_list, output, input, mask

            x = self.model_coarse_1(x)
            x = x.squeeze(2)
            coarse = self.model_coarse_2(x)

        else:
            dw_input = torch.cat([data[:, 0:1, :, :] for data in input], dim=1)
            mask_merge = (torch.mean(dw_input, dim=1, keepdim=True) == 2)    # gaps in all time
            coarse = torch.mean(torch.where(dw_input != 2, dw_input, 0), dim=1, keepdim=True)
            coarse = torch.where(mask_merge, 2, coarse)

        if not self.opt.no_stage2:
            coarse_down0 = self.dw_start(coarse)           # 256 -> 256
            coarse_down1 = self.dw_down1(coarse_down0)     # 256 -> 128
            coarse_down2 = self.dw_down2(coarse_down1)     # 128 -> 64
            coarse_down3 = self.dw_down3(coarse_down2)     # 64 -> 32
            # todo: sar_down123
            x = self.sar_start(target_S1)           # 256 -> 32
            x = self.sar_up1(x - coarse_down3)      # 32 -> 64
            x = self.sar_up2(x - coarse_down2)      # 64 -> 128
            x = self.sar_up3(x - coarse_down1)      # 128 -> 256
            x = self.sar_final(x - coarse_down0)    # 256 -> 256
            del target_S1, coarse_down0, coarse_down1, coarse_down2, coarse_down3

            x = self.final_conv(x)
            x = coarse + x
            x = self.out_activate(x)

            return x, coarse
        else:
            return coarse


class WPGCT_noDW(nn.Module):

    def __init__(self, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(WPGCT_noDW, self).__init__()

        # assert opt.n_input_samples >= 3

        self.n_samples = opt.n_input_samples
        opt.no_stage1 = False
        opt.no_stage2 = False
        self.opt = opt

        # --------------------------- coarse net ---------------------------
        self.model_initial = nn.Sequential(
            GatedConv2dWithActivation(2, opt.ngf, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(opt.ngf, 2 * opt.ngf, 3, 2, padding=get_pad(256, 3, 2)),
            GatedConv2dWithActivation(2 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample to 64
            GatedConv2dWithActivation(2 * opt.ngf, 4 * opt.ngf, 3, 2, padding=get_pad(128, 3, 2)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convolution
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),

            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample to 128
            GatedDeConv2dWithActivation(2, 4 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
            # upsample to 256
            GatedDeConv2dWithActivation(2, 2 * opt.ngf, opt.ngf, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(opt.ngf, opt.ngf, 3, 1, padding=get_pad(256, 3, 1)),
        )

        self.model_coarse_1 = nn.Sequential(
            nn.Conv3d(opt.ngf, opt.ngf, kernel_size=(opt.n_input_samples, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(opt.ngf),
            nn.ReLU(True))
        self.model_coarse_2 = nn.Sequential(
            nn.Conv2d(opt.ngf, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # --------------------------- align and refine net ---------------------------
        self.dw_start = nn.Conv2d(1, opt.embed_dim, 3, 1, 1)
        self.dw_down1 = nn.Sequential(
            # downsample to 128
            nn.Conv2d(opt.embed_dim, opt.embed_dim // 2, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
        self.dw_down2 = nn.Sequential(
            # downsample to 64
            nn.Conv2d(opt.embed_dim * 2, opt.embed_dim, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )
        self.dw_down3 = nn.Sequential(
            # downsample to 32
            nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 2, 3, 1, 1),
            nn.PixelUnshuffle(2)
        )

        self.sar_start = nn.Sequential(
            nn.Conv2d(2, opt.embed_dim, 3, 1, 1),
            TransformerBlock(dim=opt.embed_dim, num_heads=8),
            # downsample to 128
            nn.Conv2d(opt.embed_dim, opt.embed_dim // 2, 3, 1, 1),
            nn.PixelUnshuffle(2),
            TransformerBlock(dim=opt.embed_dim * 2, num_heads=8),
            # downsample to 64
            nn.Conv2d(opt.embed_dim * 2, opt.embed_dim, 3, 1, 1),
            nn.PixelUnshuffle(2),
            TransformerBlock(dim=opt.embed_dim * 4, num_heads=8),
            # downsample to 32
            nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 2, 3, 1, 1),
            nn.PixelUnshuffle(2),
            TransformerBlock(dim=opt.embed_dim * 8, num_heads=8),
        )
        self.sar_up1 = nn.Sequential(
            TransformerBlock(dim=opt.embed_dim * 8, num_heads=8),
            # upsample to 64
            nn.Conv2d(opt.embed_dim * 8, opt.embed_dim * 16, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.sar_up2 = nn.Sequential(
            TransformerBlock(dim=opt.embed_dim * 4, num_heads=8),
            # upsample to 128
            nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 8, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.sar_up3 = nn.Sequential(
            TransformerBlock(dim=opt.embed_dim * 2, num_heads=8),
            # upsample to 256
            nn.Conv2d(opt.embed_dim * 2, opt.embed_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.sar_final = TransformerBlock(dim=opt.embed_dim, num_heads=8)

        self.final_conv = nn.Conv2d(opt.embed_dim, 1, 3, 1, 1)
        self.out_activate = nn.Sigmoid()

    def forward(self, input, **kwargs):
        """Standard forward"""
        target_S1 = None
        if 'B_S1' in kwargs.keys():
            target_S1 = kwargs['B_S1']

        initial_list = []
        for t in range(self.n_samples):
            output = self.model_initial(input[t][:, 1:, :, :])
            initial_list.append(output)

        x = torch.stack(initial_list, dim=2)
        del initial_list, output, input

        x = self.model_coarse_1(x)
        x = x.squeeze(2)
        coarse = self.model_coarse_2(x)

        coarse_down0 = self.dw_start(coarse)  # 256 -> 256
        coarse_down1 = self.dw_down1(coarse_down0)  # 256 -> 128
        coarse_down2 = self.dw_down2(coarse_down1)  # 128 -> 64
        coarse_down3 = self.dw_down3(coarse_down2)  # 64 -> 32

        x = self.sar_start(target_S1)  # 256 -> 32
        x = self.sar_up1(x - coarse_down3)  # 32 -> 64
        x = self.sar_up2(x - coarse_down2)  # 64 -> 128
        x = self.sar_up3(x - coarse_down1)  # 128 -> 256
        x = self.sar_final(x - coarse_down0)  # 256 -> 256
        del target_S1, coarse_down0, coarse_down1, coarse_down2, coarse_down3

        x = self.final_conv(x)
        x = coarse + x
        x = self.out_activate(x)

        return x, coarse


class WPGCT_noS1(nn.Module):

    def __init__(self, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        super(WPGCT_noS1, self).__init__()

        # assert opt.n_input_samples >= 3

        self.n_samples = opt.n_input_samples
        opt.no_stage1 = False
        opt.no_stage2 = False
        self.opt = opt

        # --------------------------- coarse net ---------------------------
        self.model_initial = nn.Sequential(
            GatedConv2dWithActivation(2, opt.ngf, 5, 1, padding=get_pad(256, 5, 1)),
            # downsample 128
            GatedConv2dWithActivation(opt.ngf, 2 * opt.ngf, 3, 2, padding=get_pad(256, 3, 2)),
            GatedConv2dWithActivation(2 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
            # downsample to 64
            GatedConv2dWithActivation(2 * opt.ngf, 4 * opt.ngf, 3, 2, padding=get_pad(128, 3, 2)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            # atrous convolution
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=2, padding=get_pad(64, 3, 1, 2)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=4, padding=get_pad(64, 3, 1, 4)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=8, padding=get_pad(64, 3, 1, 8)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, dilation=16, padding=get_pad(64, 3, 1, 16)),

            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            GatedConv2dWithActivation(4 * opt.ngf, 4 * opt.ngf, 3, 1, padding=get_pad(64, 3, 1)),
            # upsample to 128
            GatedDeConv2dWithActivation(2, 4 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
            GatedConv2dWithActivation(2 * opt.ngf, 2 * opt.ngf, 3, 1, padding=get_pad(128, 3, 1)),
            # upsample to 256
            GatedDeConv2dWithActivation(2, 2 * opt.ngf, opt.ngf, 3, 1, padding=get_pad(256, 3, 1)),
            GatedConv2dWithActivation(opt.ngf, opt.ngf, 3, 1, padding=get_pad(256, 3, 1)),
        )

        self.model_coarse_1 = nn.Sequential(
            nn.Conv3d(opt.ngf, opt.ngf, kernel_size=(opt.n_input_samples, 3, 3), padding=(0, 1, 1)),
            nn.BatchNorm3d(opt.ngf),
            nn.ReLU(True))
        self.model_coarse_2 = nn.Sequential(
            nn.Conv2d(opt.ngf, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # --------------------------- align and refine net ---------------------------
        self.dw_start = nn.Sequential(
            nn.Conv2d(1, opt.embed_dim, 3, 1, 1),
            TransformerBlock(dim=opt.embed_dim, num_heads=8)
        )
        self.dw_down1 = nn.Sequential(
            # downsample to 128
            nn.Conv2d(opt.embed_dim, opt.embed_dim // 2, 3, 1, 1),
            nn.PixelUnshuffle(2),
            TransformerBlock(dim=opt.embed_dim * 2, num_heads=8),
        )
        self.dw_down2 = nn.Sequential(
            # downsample to 64
            nn.Conv2d(opt.embed_dim * 2, opt.embed_dim, 3, 1, 1),
            nn.PixelUnshuffle(2),
            TransformerBlock(dim=opt.embed_dim * 4, num_heads=8),
        )
        self.dw_down3 = nn.Sequential(
            # downsample to 32
            nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 2, 3, 1, 1),
            nn.PixelUnshuffle(2),
            TransformerBlock(dim=opt.embed_dim * 8, num_heads=8)
        )
        self.dw_up1 = nn.Sequential(
            TransformerBlock(dim=opt.embed_dim * 8, num_heads=8),
            # upsample to 64
            nn.Conv2d(opt.embed_dim * 8, opt.embed_dim * 16, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.dw_up2 = nn.Sequential(
            TransformerBlock(dim=opt.embed_dim * 4, num_heads=8),
            # upsample to 128
            nn.Conv2d(opt.embed_dim * 4, opt.embed_dim * 8, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.dw_up3 = nn.Sequential(
            TransformerBlock(dim=opt.embed_dim * 2, num_heads=8),
            # upsample to 256
            nn.Conv2d(opt.embed_dim * 2, opt.embed_dim * 4, 3, 1, 1),
            nn.PixelShuffle(2)
        )
        self.dw_final = TransformerBlock(dim=opt.embed_dim, num_heads=8)

        self.final_conv = nn.Conv2d(opt.embed_dim, 1, 3, 1, 1)
        self.out_activate = nn.Sigmoid()

    def forward(self, input, **kwargs):
        """Standard forward"""
        mask = None
        if 'mask' in kwargs.keys():
            mask = kwargs['mask']

        initial_list = []
        for t in range(self.n_samples):
            output = self.model_initial(torch.cat([input[t][:, 0:1, :, :], mask[t]], dim=1))
            initial_list.append(output)

        x = torch.stack(initial_list, dim=2)
        del initial_list, output, input, mask

        x = self.model_coarse_1(x)
        x = x.squeeze(2)
        coarse = self.model_coarse_2(x)

        coarse_down0 = self.dw_start(coarse)  # 256 -> 256
        coarse_down1 = self.dw_down1(coarse_down0)  # 256 -> 128
        coarse_down2 = self.dw_down2(coarse_down1)  # 128 -> 64
        coarse_down3 = self.dw_down3(coarse_down2)  # 64 -> 32

        x = self.dw_up1(coarse_down3)  # 32 -> 64
        x = self.dw_up2(x + coarse_down2)  # 64 -> 128
        x = self.dw_up3(x + coarse_down1)  # 128 -> 256
        x = self.dw_final(x + coarse_down0)  # 256 -> 256
        del coarse_down0, coarse_down1, coarse_down2, coarse_down3

        x = self.final_conv(x)
        x = coarse + x
        x = self.out_activate(x)

        return x, coarse


class SNDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf):
        super(SNDiscriminator, self).__init__()
        cnum = ndf
        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(1, cnum, 5, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(cnum, 2 * cnum, 5, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(2 * cnum, 4 * cnum, 5, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(4 * cnum, 4 * cnum, 5, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(4 * cnum, 4 * cnum, 5, 2, padding=get_pad(16, 5, 2)),
            SNConvWithActivation(4 * cnum, 4 * cnum, 5, 2, padding=get_pad(8, 5, 2)),
        )

    def forward(self, input):
        x = self.discriminator_net(input)
        x = x.view((x.size(0), -1))
        return x


class MultiBranchSNDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf, n_samples):
        super(MultiBranchSNDiscriminator, self).__init__()

        self.n_samples = n_samples

        self.discriminator_net = nn.Sequential(
            SNConvWithActivation(input_nc, ndf, 5, 2, padding=get_pad(256, 5, 2)),
            SNConvWithActivation(ndf, 2 * ndf, 5, 2, padding=get_pad(128, 5, 2)),
            SNConvWithActivation(2 * ndf, 4 * ndf, 5, 2, padding=get_pad(64, 5, 2)),
            SNConvWithActivation(4 * ndf, 4 * ndf, 5, 2, padding=get_pad(32, 5, 2)),
            SNConvWithActivation(4 * ndf, 4 * ndf, 5, 2, padding=get_pad(16, 5, 2)),
        )

        self.aggregate_net_1 = SNConvWithActivation3D(4 * ndf, 4 * ndf, 3, stride=1, padding=(0, 1, 1))

        self.aggregate_net_2 = SNConvWithActivation(4 * ndf, 4 * ndf, 5, 2, padding=get_pad(8, 5, 2))

    def forward(self, input):
        initial_list = []
        for t in range(self.n_samples):
            output = self.discriminator_net(input[t])
            initial_list.append(output)
        x = torch.stack(initial_list, dim=2)
        del initial_list, output, input

        x = self.aggregate_net_1(x)
        x = x.squeeze(2)
        x = self.aggregate_net_2(x)

        x = x.view((x.size(0), -1))
        return x


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.InstanceNorm2d
        else:
            use_bias = norm_layer != nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


class SNConvWithActivation(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation, self).__init__()
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv2d = torch.nn.utils.spectral_norm(self.conv2d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv2d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return x


class SNConvWithActivation3D(torch.nn.Module):
    """
    SN convolution for spetral normalization conv
    """

    def __init__(self, in_channels, out_channels, kernel_size, padding, stride=1, dilation=1, groups=1, bias=True,
                 activation=torch.nn.LeakyReLU(0.2, inplace=True)):
        super(SNConvWithActivation3D, self).__init__()
        self.conv3d = torch.nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv3d = torch.nn.utils.spectral_norm(self.conv3d)
        self.activation = activation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def forward(self, input):
        x = self.conv3d(input)
        if self.activation is not None:
            return self.activation(x)
        else:
            return