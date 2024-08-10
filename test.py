import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
from util import pytorch_ssim, html
import tifffile as tif


def compute_metric(real_B, fake_B):
    rmse = torch.sqrt(torch.mean(torch.square(real_B - fake_B)))

    psnr = 20 * torch.log10(1 / rmse)

    mae = torch.mean(torch.abs(real_B - fake_B))

    mat = real_B * fake_B
    mat = torch.sum(mat)
    mat = torch.div(mat, torch.sqrt(torch.sum(real_B * real_B)))
    mat = torch.div(mat, torch.sqrt(torch.sum(fake_B * fake_B)))
    sam = torch.mean(torch.acos(torch.clamp(mat, -1, 1)))

    ssim = pytorch_ssim.ssim(real_B, fake_B)

    return {'RMSE': rmse.cpu().numpy().item(),
            'MAE': mae.cpu().numpy().item(),
            'PSNR': psnr.cpu().numpy().item(),
            'SAM': np.rad2deg(sam.cpu().numpy().item()),
            'SSIM': ssim.cpu().numpy().item()}


def save_metric(metric, path, label, epoch, patch_list, mode='all'):
    f = open(path + f'/test_metric_{label}_{epoch}_{mode}.txt', 'w')

    f.write('Patch'.center(20))
    # for each metric, save the item name
    for batch_metric in metric:
        for idx, (crit, value) in enumerate(batch_metric.items()):
            f.write('\t' + str(crit).center(20))
        break
    f.write('\n')

    # for each metric, compute the item value
    m = []
    for p, batch_metric in enumerate(metric):
        f.write(str(patch_list[p]).center(20))
        dum = []
        # iterate over each item
        for idx, (crit, value) in enumerate(batch_metric.items()):
            f.write('\t' + '{:.5f}'.format(value).center(20))
            dum.append(value)
        f.write('\n')
        m.append(dum)

    f.write('\n\n')
    f.write('Patch'.center(20))
    for batch_metric in metric:
        for idx, (crit, value) in enumerate(batch_metric.items()):
            f.write('\t' + str(crit).center(20))
        break

    f.write('\n')
    f.write('Overall mean'.center(20))
    # for each metric, compute the average value
    for each in np.nanmean(m, axis=0):
        f.write('\t' + '{:.5f}'.format(each).center(20))

    f.close()


def save_images(visuals, image_dir, opt, patch):
    # image_dir = webpage.get_image_dir()
    # webpage.add_header('patch_{}'.format(patch) + '\t----' + opt.name)
    image_dir = os.path.join(image_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    ims, txts = [], []
    for label, im_data in visuals.items():
        if 'input' in label:
            input_dir = os.path.join(opt.dataroot, 'DW', label.split('_')[0])
            img_path = os.path.join(input_dir, '{}.tif'.format(patch))
        else:
            image_name = '%s_%s.tif' % (patch, label)
            img_path = os.path.join(image_dir, image_name)
            tif.imwrite(img_path, im_data)
        ims.append(img_path)
        txts.append(label)

    # webpage.add_images(ims, txts, width=256)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.serial_batches = True       # disable data shuffling
    opt.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1             # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()  # test with eval mode. This affects layers like batchnorm and dropout.

    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch))  # define the website directory
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))

    model_metric_30 = []
    model_metric_60 = []
    model_metric_100 = []
    out_dir_30 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_30')
    out_dir_60 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_60')
    out_dir_100 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_100')
    patch_list_30 = []
    patch_list_60 = []
    patch_list_100 = []
    model_metric = []
    out_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_all')
    patch_list = []

    # if opt.various_gap_coverage:
    #     model_metric_30 = []
    #     model_metric_60 = []
    #     model_metric_100 = []
    #     out_dir_30 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_30')
    #     out_dir_60 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_60')
    #     out_dir_100 = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_100')
    #     patch_list_30 = []
    #     patch_list_60 = []
    #     patch_list_100 = []
    # else:
    #     model_metric = []
    #     if include_simple_baselines:
    #         base_leastcloudy_metric = []
    #         base_mosaic_metric = []
    #     out_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.epoch) + '_all')
    #     patch_list = []

    # 测试时，对于模型输入的gap coverage，选取input[0]，即时间最邻近的输入，进行计算
    # 对于Least方法，则使用三个input中最小gap coverage
    for i, data in enumerate(dataset):
        if i % 10 == 0:
            print('processing %04d-th batch' % (i + 1), flush=True)

        model.set_input(data)  # unpack data from data loader
        model.test()           # run inference

        # compute least cloudy and mosaicing baseline
        for j in range(opt.batch_size):
            # process our model
            if 'refine' in opt.netG:
                fake = model.fake_B[1]
            elif opt.use_aux and not opt.no_stage2:
                fake = model.fake_B[0]
            else:
                fake = model.fake_B

            # least
            # least_cloudy_idx = np.argsort([torch.sum(model.A_mask[k][j]).cpu().numpy() for k in range(opt.n_input_samples)])[0]
            # fake = model.DW_input[least_cloudy_idx]
            # fake[fake == 2] = 0.5

            # mosaic
            # mosaic = []
            # for k in range(opt.n_input_samples):
            #     masked_t = np.where(model.A_mask[k][j].cpu().numpy(), np.nan, model.DW_input[k][j].cpu().numpy())
            #     mosaic.append(masked_t)
            # # apply mean mosaicing
            # mosaic = torch.tensor(np.nanmean(np.array(mosaic), 0, keepdims=True))
            # # for pixels that are nan across all time points: take neutral value
            # mosaic[torch.isnan(mosaic)] = 0.5
            # fake = mosaic.to(model.device)

            if j > len(fake) - 1:
                break

            visual_dict = {'input1_DW': model.DW_input[0][j].cpu().numpy(),
                           # 'input2_DW': model.DW_input[1][j].cpu().numpy(),
                           # 'input3_DW': model.DW_input[2][j].cpu().numpy(),
                           'model_output': fake[j].cpu().numpy(),
                           'target': model.real_B[j].cpu().numpy()}

            metric = compute_metric(model.real_B[j], fake[j])

            model_metric.append(metric)
            save_images(visual_dict, out_dir, opt, model.patch_id[j].cpu().numpy())
            patch_list.append(model.patch_id[j].cpu().numpy())

            coverage = np.mean(model.A_mask[0][j].cpu().numpy())
            if 0 < coverage <= 0.3:
                model_metric_30.append(metric)
                save_images(visual_dict, out_dir_30, opt, model.patch_id[j].cpu().numpy())
                patch_list_30.append(model.patch_id[j].cpu().numpy())
            elif 0.3 < coverage <= 0.6:
                model_metric_60.append(metric)
                save_images(visual_dict, out_dir_60, opt, model.patch_id[j].cpu().numpy())
                patch_list_60.append(model.patch_id[j].cpu().numpy())
            else:
                model_metric_100.append(metric)
                save_images(visual_dict, out_dir_100, opt, model.patch_id[j].cpu().numpy())
                patch_list_100.append(model.patch_id[j].cpu().numpy())

            # if opt.various_gap_coverage:
            #     visual_dict = {'input1_DW': model.DW_input[0][j].cpu().numpy(),
            #                    # 'input2_DW': model.DW_input[1][j].cpu().numpy(),
            #                    # 'input3_DW': model.DW_input[2][j].cpu().numpy(),
            #                    'model_output': fake[j].cpu().numpy(),
            #                    'target': model.real_B[j].cpu().numpy()}
            #
            #     coverage = np.mean(model.A_mask[0][j].cpu().numpy())
            #     if 0 < coverage <= 0.3:
            #         model_metric_30.append(metric)
            #         save_images(visual_dict, out_dir_30, opt, model.patch_id[j].cpu().numpy())
            #         patch_list_30.append(model.patch_id[j].cpu().numpy())
            #     elif 0.3 < coverage <= 0.6:
            #         model_metric_60.append(metric)
            #         save_images(visual_dict, out_dir_60, opt, model.patch_id[j].cpu().numpy())
            #         patch_list_60.append(model.patch_id[j].cpu().numpy())
            #     else:
            #         model_metric_100.append(metric)
            #         save_images(visual_dict, out_dir_100, opt, model.patch_id[j].cpu().numpy())
            #         patch_list_100.append(model.patch_id[j].cpu().numpy())
            #
            # else:
            #     model_metric.append(metric)
            #

                # visual_dict = {'input1_DW': model.DW_input[0][j].cpu().numpy(),
                #                'input2_DW': model.DW_input[1][j].cpu().numpy(),
                #                'input3_DW': model.DW_input[2][j].cpu().numpy(),
                #                'model_output': fake[j].cpu().numpy(),
                #                'target': model.real_B[j].cpu().numpy()}
            #
            #     save_images(visual_dict, out_dir, opt, model.patch_id[j].cpu().numpy())
            #     patch_list.append(model.patch_id[j].cpu().numpy())

    # webpage.save()  # save the HTML
    # save metric stats for the STGAN model
    if len(model_metric_30) > 0:
        save_metric(model_metric_30, out_dir_30, opt.name, opt.epoch, patch_list_30, '30')
    if len(model_metric_30) > 0:
        save_metric(model_metric_60, out_dir_60, opt.name, opt.epoch, patch_list_60, '60')
    if len(model_metric_100) > 0:
        save_metric(model_metric_100, out_dir_100, opt.name, opt.epoch, patch_list_100, '100')
    save_metric(model_metric, out_dir, opt.name, opt.epoch, patch_list, 'all')
    # if opt.various_gap_coverage:
    #     if len(model_metric_30) > 0:
    #         save_metric(model_metric_30, out_dir_30, opt.name, opt.epoch, patch_list_30, '30')
    #     if len(model_metric_30) > 0:
    #         save_metric(model_metric_60, out_dir_60, opt.name, opt.epoch, patch_list_60, '60')
    #     if len(model_metric_100) > 0:
    #         save_metric(model_metric_100, out_dir_100, opt.name, opt.epoch, patch_list_100, '100')
    # else:
    #     save_metric(model_metric, out_dir, opt.name, opt.epoch, patch_list, 'all')
        # if include_simple_baselines:
        #     save_metric(base_leastcloudy_metric, out_dir, 'leastcloudy', opt.epoch, patch_list)
        #     save_metric(base_mosaic_metric, out_dir, 'mosaic', opt.epoch, patch_list)
