import argparse
import os
from util import util
import torch
import models
import data


class BaseOptions():
    """This class defines options used during both training and test time.

    It also implements several helper functions such as parsing, printing, and saving the options.
    It also gathers additional options defined in <modify_commandline_options> functions in both dataset class and model class.
    """

    def __init__(self):
        """Reset the class; indicates the class hasn't been initailized"""
        self.initialized = False

    def initialize(self, parser):
        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--dataroot', required=False, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)',
                            default='G:\\reconstruction\\data\\Samples\\DW_S1_dataset_nearest_input_new')
        parser.add_argument('--name', type=str, default='exemplary_training_run', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, help='models are saved here',
                            default='G:\\reconstruction\\result')

        # model parameters
        parser.add_argument('--model', type=str, default='temporal_branched',
                            help='chooses which model to use. [cycle_gan | pix2pix | test | colorization]')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='n_layers',
                            help='specify discriminator architecture [basic | n_layers | pixel | sn]. The basic model '
                                 'is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--netG', type=str, default='resnet',
                            help='specify generator architecture [resnet | gated_resnet | dilated_gated_resnet]')
        parser.add_argument('--n_blocks', type=int, default=4, help='number of residual blocks in Generator')
        parser.add_argument('--use_attention', action='store_true', help='whether to use attention module in Generator')
        parser.add_argument('--use_s1_constrain', action='store_true', help='whether to use s1 constrain in Generator')
        parser.add_argument('--use_mask', action='store_true', help='whether to use mask in Generator')
        parser.add_argument('--embed_dim', type=int, default=64,
                            help='If using Transformer, specify number of embed dims')
        parser.add_argument('--norm', type=str, default='instance',
                            help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal',
                            help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--use_perceptual_loss', action='store_true', help='use perceptual loss in training')
        parser.add_argument('--vgg16_path', type=str, default=r'G:\reconstruction\data\pretrained\vgg16_13C.pth',
                            help='the path of pretrained VGG16 network')
        parser.add_argument('--lambda_percep', type=float, default=1., help='weight of the perceptual loss in training')
        parser.add_argument('--layers_percep', type=str, default='original',
                            help='layers of VGG16 to use for the perceptual loss (choose: dip, video, original, experimental)')
        parser.add_argument('--use_aux', action='store_true', help='whether to use aux loss')
        parser.add_argument('--lambda_aux', type=float, default=0.1, help='weight of aux loss in training')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--no_stage1', action='store_true', help='no stage1 for the generator')
        parser.add_argument('--no_stage2', action='store_true', help='no stage2 for the generator')

        # dataset parameters
        parser.add_argument('--sample_type', type=str, default='nearest_time',
                            help='choose the format of input data. [nearest_time]')
        parser.add_argument('--input_type', type=str, default='train',
                            help='choose the type of input. [all | test | val | train]')
        parser.add_argument('--dataset_mode', type=str, default='dw_s1',
                            help='chooses how datasets are loaded. [dw_s1]')
        parser.add_argument('--s1_channels', type=str, default='vv+vh',
                            help='chooses which s1 channel to use. [vv | vh | vv+vh]')
        parser.add_argument('--n_input_samples', default=3, type=int, help='number of input samples')
        parser.add_argument('--serial_batches', action='store_true',
                            help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--max_dataset_size', type=int, default=1,
                            help='Maximum number of samples allowed per dataset. If the dataset directory contains '
                                 'more than max_dataset_size, only a subset is loaded.')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop',
                            help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | '
                                 'scale_width_and_crop | none]')
        parser.add_argument('--no_flip', action='store_true',
                            help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--no_dw', action='store_true',
                            help='if specified, do not use DW in input')
        parser.add_argument('--no_s1', action='store_true',
                            help='if specified, do not use S1 in input')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest',
                            help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0',
                            help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str,
                            help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--resnet_F', type=int, default=256, help='If using ResNet, specify number of feature maps F')
        parser.add_argument('--resnet_B', type=int, default=16, help='If using ResNet, specify number of feature maps F')
        parser.add_argument('--no_64C', action='store_true', help='dont use the intermediate reduction to 64 channels')
        parser.add_argument('--display_winsize', type=int, default=256,
                            help='display window size for both visdom and HTML')
        parser.add_argument('--use_amp', action='store_true', help='whether to use mix precision training')

        # cloud coverage
        parser.add_argument('--min_cov', type=float, default=0.1, help='minimum acceptable cloud coverage')
        parser.add_argument('--max_cov', type=float, default=0.9, help='maximum acceptable cloud coverage')

        self.initialized = True
        return parser

    def gather_options(self):
        """Initialize our parser with basic options(only once).
        Add additional model-specific and dataset-specific options.
        These options are defined in the <modify_commandline_options> function
        in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)
        opt, _ = parser.parse_known_args()  # parse again with new defaults

        # modify dataset-related parser options
        dataset_name = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_name)
        parser = dataset_option_setter(parser, self.isTrain)

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def print_options(self, opt, phase):
        """Print and save options

        It will print both current options and default values(if different).
        It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        # print(message)

        # save to the disk
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, phase + '_opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test
        if opt.isTrain:
            phase = 'Train'
        else:
            phase = 'Test'

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt, phase)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt
