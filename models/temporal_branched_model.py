import torch
from torch.cuda.amp import autocast
from .base_model import BaseModel
from . import networks_branched as networks
from util import util
import warnings


class TemporalBranchedModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        # parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        parser.set_defaults(norm='batch')
        parser.add_argument('--scramble', type=bool, default=False, help='scramble order of input images?')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=1.0, help='weight for L1 loss')
            parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight of GAN loss for generator and discriminator')
        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)

        # specify the images you want to save/display.
        # The training/test scripts will call <BaseModel.get_current_visuals>
        self.opt = opt
        if opt.netG == 'dsen2cr':
            self.opt.lambda_GAN = 0
            self.n_input_samples = 1
            self.use_perceptual_loss = False

        self.visual_names = []
        for i in range(opt.n_input_samples):
            self.visual_names.append(f'real_A_{i}')
            self.visual_names.append(f'A_{i}_S1')
            self.visual_names.append(f'A_{i}_mask')
            self.visual_names += ['real_B_S1']
        self.visual_names = self.visual_names + ['real_B', 'fake_B']

        # specify the models you want to save to the disk.
        # The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain and opt.lambda_GAN != 0:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        if self.opt.s1_channels in ['vv', 'vh']:
            self.input_nc = 2
        else:
            self.input_nc = 3

        # define a generator
        self.netG = networks.define_G(self.device, self.gpu_ids, opt)

        if self.isTrain:
            # define loss functions
            if opt.lambda_GAN != 0:       # if not 0 -> include GAN loss and discriminator
                # define a discriminator
                self.netD = networks.define_D(2, opt, self.gpu_ids)
                self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
                # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_D)
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

        if opt.use_perceptual_loss and self.opt.lambda_percep != 0:
            # specify the training losses you want to print out.
            # The training/test scripts will call <BaseModel.get_current_losses>
            assert not opt.vgg16_path == 'none', 'Missing input of VGG16 path.'

            if self.isTrain and opt.lambda_GAN != 0:
                self.loss_names = ['G_GAN', 'G_loss', 'perceptual', 'G', 'D_real', 'D_fake', 'D']
            else:
                self.loss_names = ['G_loss', 'perceptual', 'G']

            # specify channels to compute perceptual loss over, e.g.
            # [11,20,29] as in DIP code (relu3_1, relu4_1, relu5_1) https://github.com/DmitryUlyanov/deep-image-prior/blob/master/utils/perceptual_loss/perceptual_loss.py
            # ['3', '8', '15'] as in internal video learning ('3': "relu1_2",  '8': "relu2_2", '15': "relu3_3", '22': "relu4_3") https://github.com/Haotianz94/IL_video_inpainting/blob/7bf67772b19f44245495c18f79002fea5853bb57/src/configs/base.py & https://github.com/Haotianz94/IL_video_inpainting/blob/7bf67772b19f44245495c18f79002fea5853bb57/src/models/perceptual.py,
            # ['3', '8', '15', '22'] as in original Gatys et al paper (relu1_2, relu2_2, relu3_3, and relu4_3)
            # [8, 15, 22, 29] is also an option to give a try
            # --> labels correspond to one another

            perceptual_layers = {'dip': [11, 20, 29],
                                 'video': [3, 8, 15],
                                 'original': [3, 8, 15, 22],
                                 'experimental': [8, 15, 22, 29]}

            self.netL = util.LossNetwork(opt.vgg16_path, perceptual_layers[opt.layers_percep], self.device)
        else:
            if self.isTrain and opt.lambda_GAN != 0:
                self.loss_names = ['G_GAN', 'G_loss', 'G', 'D_real', 'D_fake', 'D']
            else:
                self.loss_names = ['G_loss', 'G']
        self.total_iters = 0

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.
        """

        # dynamically process a variable number of input time points
        A_input, self.A_mask = [], []
        for i in range(self.opt.n_input_samples):
            self.A_mask.append(input['A_mask'][i].to(self.device))
            setattr(self, f'A_{i}_mask', self.A_mask[i])
            A_input.append(input['A_DW'][i].to(self.device))
            setattr(self, f'real_A_{i}', A_input[i])

        self.real_A_input, self.A_S1 = [], []
        for i in range(self.opt.n_input_samples):
            S1 = input['A_S1'][i].to(self.device)
            setattr(self, f'A_{i}_S1', S1)
            self.real_A_input.append(torch.cat((A_input[i], S1), 1).to(self.device))
            self.A_S1.append(S1)

        # bookkeeping of target cloud-free patch
        self.real_B = input['B'].to(self.device)
        # bookkeeping of target mask
        self.real_B_S1 = input['B_S1']

        self.DW_input = A_input

        self.patch_id = input['patch_id']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test> (in base_model.py)."""
        # gated conv needs concat mask
        args = {}
        if self.opt.use_mask:
            args['mask'] = self.A_mask
        if self.opt.use_s1_constrain:
            args['B_S1'] = self.real_B_S1

        if self.opt.use_amp:
            with autocast():
                self.fake_B = self.netG(self.real_A_input, **args)
        else:
            self.fake_B = self.netG(self.real_A_input, **args)

    def backward_D(self, scaler_D):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        if self.opt.use_aux and not self.opt.no_stage2:
            fake = self.fake_B[0].detach()
        else:
            fake = self.fake_B.detach()

        if self.opt.use_amp:
            with autocast():
                # fake
                if self.opt.netD == 'mbsn':
                    D_fake = [torch.cat((fake, self.A_mask[t]), 1) for t in range(self.opt.n_input_samples)]
                else:
                    D_fake = fake
                pred_fake = self.netD(D_fake)
                # real
                if self.opt.netD == 'mbsn':
                    D_real = [torch.cat((self.real_B, self.A_mask[t]), 1) for t in range(self.opt.n_input_samples)]
                else:
                    D_real = self.real_B
                pred_real = self.netD(D_real)
                # loss
                self.loss_D_fake = self.criterionGAN(pred_fake, False)
                self.loss_D_real = self.criterionGAN(pred_real, True)
                self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN
            scaler_D.scale(self.loss_D).backward()
        else:
            # fake
            if self.opt.netD == 'mbsn':
                D_fake = [torch.cat((fake, self.A_mask[t]), 1) for t in range(self.opt.n_input_samples)]
            else:
                D_fake = fake
            pred_fake = self.netD(D_fake)
            # real
            if self.opt.netD == 'mbsn':
                D_real = [torch.cat((self.real_B, self.A_mask[t]), 1) for t in range(self.opt.n_input_samples)]
            else:
                D_real = self.real_B
            pred_real = self.netD(D_real)
            # loss
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5 * self.opt.lambda_GAN
            self.loss_D.backward()

        return scaler_D

    def criterionG(self, fake_B, real_B):
        if self.opt.G_loss == 'L1':
            loss = torch.nn.L1Loss()
            return loss(fake_B, real_B)
        else:
            raise Exception("Undefined G loss type.")

    def get_perceptual_loss(self):
        loss = 0.
        if self.opt.use_aux and not self.opt.no_stage2:
            fake = self.netL(self.fake_B[0])
        else:
            fake = self.netL(self.fake_B)

        real = self.netL(self.real_B)

        mse = torch.nn.MSELoss()
        for i in range(len(fake)):
            loss += mse(fake[i], real[i])
        return loss

    def carl_error(self, true, pred, input, mask, lambada=1.0):
        """Computes the Cloud-Adaptive Regularized Loss (CARL)"""
        clearmask = torch.ones_like(mask) - mask

        cscmae = torch.mean(clearmask * torch.abs(pred - input) + mask * torch.abs(pred - true)) + \
                 lambada * torch.mean(torch.abs(pred - true))

        return cscmae

    def backward_G(self, scaler_G):
        """Calculate GAN and L1 loss for the generator"""
        if self.opt.lambda_GAN != 0:
            # First, G(A) should fake the discriminator
            if self.opt.use_aux and not self.opt.no_stage2:
                fake = self.fake_B[0]
            else:
                fake = self.fake_B

            if self.opt.use_amp:
                with autocast():
                    # fake
                    if self.opt.netD == 'mbsn':
                        D_fake = [torch.cat((fake, self.A_mask[t]), 1) for t in range(self.opt.n_input_samples)]
                    else:
                        D_fake = fake
                    pred_fake = self.netD(D_fake)
                    # loss
                    self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.5
            else:
                # fake
                if self.opt.netD == 'mbsn':
                    D_fake = [torch.cat((fake, self.A_mask[t]), 1) for t in range(self.opt.n_input_samples)]
                else:
                    D_fake = fake
                pred_fake = self.netD(D_fake)
                # loss
                self.loss_G_GAN = self.criterionGAN(pred_fake, True) * 0.5

        # Second, G(A) = B
        if self.opt.use_amp:
            with autocast():
                if self.opt.netG == 'dsen2cr':
                    self.loss_G_loss = self.carl_error(self.real_B, self.fake_B, self.DW_input[0], self.A_mask[0])
                elif self.opt.use_aux and not self.opt.no_stage2:
                    if not self.opt.no_stage1:
                        self.loss_G_loss = self.criterionG(self.fake_B[0], self.real_B) \
                                           + self.opt.lambda_aux * self.criterionG(self.fake_B[1], self.real_B)
                    else:
                        self.loss_G_loss = self.criterionG(self.fake_B[0], self.real_B)
                else:
                    self.loss_G_loss = self.criterionG(self.fake_B, self.real_B)
                # combine loss and calculate gradients
                if self.opt.use_perceptual_loss and self.opt.lambda_percep != 0:
                    self.loss_perceptual = self.get_perceptual_loss()
                    if self.opt.lambda_GAN != 0:
                        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + self.opt.lambda_L1 * self.loss_G_loss \
                                      + self.opt.lambda_percep * self.loss_perceptual
                    else:
                        self.loss_G = self.opt.lambda_L1 * self.loss_G_loss + self.opt.lambda_percep * self.loss_perceptual
                else:
                    if self.opt.lambda_GAN != 0:
                        self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + self.opt.lambda_L1 * self.loss_G_loss
                    else:
                        self.loss_G = self.opt.lambda_L1 * self.loss_G_loss
            scaler_G.scale(self.loss_G).backward()
        else:
            if self.opt.netG == 'dsen2cr':
                self.loss_G_loss = self.carl_error(self.real_B, self.fake_B, self.DW_input[0], self.A_mask[0])
            elif self.opt.use_aux and not self.opt.no_stage2:
                if not self.opt.no_stage1:
                    self.loss_G_loss = self.criterionG(self.fake_B[0], self.real_B) \
                                       + self.opt.lambda_aux * self.criterionG(self.fake_B[1], self.real_B)
                else:
                    self.loss_G_loss = self.criterionG(self.fake_B[0], self.real_B)
            else:
                self.loss_G_loss = self.criterionG(self.fake_B, self.real_B)
            # combine loss and calculate gradients
            if self.opt.use_perceptual_loss and self.opt.lambda_percep != 0:
                self.loss_perceptual = self.get_perceptual_loss()
                if self.opt.lambda_GAN != 0:
                    self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + self.opt.lambda_L1 * self.loss_G_loss \
                                  + self.opt.lambda_percep * self.loss_perceptual
                else:
                    self.loss_G = self.opt.lambda_L1 * self.loss_G_loss + self.opt.lambda_percep * self.loss_perceptual
            else:
                if self.opt.lambda_GAN != 0:
                    self.loss_G = self.opt.lambda_GAN * self.loss_G_GAN + self.opt.lambda_L1 * self.loss_G_loss
                else:
                    self.loss_G = self.opt.lambda_L1 * self.loss_G_loss
            self.loss_G.backward()

        return scaler_G

    def valid_grad(self, net):
        valid_gradients = True
        for name, param in net.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            warnings.warn(f'detected inf or nan values in gradients. not updating model parameters')
        return valid_gradients

    def optimize_parameters(self, scaler_G=None, scaler_D=None):
        self.forward()                   # compute fake images: G(A)
        if self.opt.lambda_GAN != 0:
            # update D
            self.set_requires_grad(self.netD, True)  # enable backprop for D
            self.optimizer_D.zero_grad()     # set D's gradients to zero
            scaler_D = self.backward_D(scaler_D)                # calculate gradients for D
            torch.nn.utils.clip_grad_norm_(self.netD.parameters(), max_norm=1.0, norm_type=2)
            if scaler_D is not None:
                scaler_D.step(self.optimizer_D)
                scaler_D.update()
            else:
                if self.valid_grad(self.netD):
                    self.optimizer_D.step()          # update D's weights
                else:
                    self.optimizer_D.zero_grad()  # do not update D
            self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G

        # update G
        # torch.autograd.set_detect_anomaly(True)
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        scaler_G = self.backward_G(scaler_G)        # calculate graidents for G
        torch.nn.utils.clip_grad_norm_(self.netG.parameters(), max_norm=1.0, norm_type=2)
        if scaler_G is not None:
            # with torch.autograd.detect_anomaly():
            scaler_G.step(self.optimizer_G)
            scaler_G.update()
        else:
            if self.valid_grad(self.netG):
                self.optimizer_G.step()  # update G's weights
            else:
                self.optimizer_G.zero_grad()  # do not update G

        return scaler_G, scaler_D
