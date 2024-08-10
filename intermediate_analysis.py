import os
import numpy as np
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
import torch
from util import pytorch_ssim, html
import tifffile as tif
import rasterio
from matplotlib import cm


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


def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    arr = tif.read().astype(np.float32)
    return arr


def get_data(opt):
    target_dw_path = os.path.join(opt.dataroot, 'DW', 'target', '{}.tif'.format(opt.test_id))
    target_s1_path = os.path.join(opt.dataroot, 'S1', 'target', '{}.tif'.format(opt.test_id))
    target_s1, target_dw = read_tif(target_s1_path), read_tif(target_dw_path)

    input_s1, input_dw, input_masks = [], [], []
    for i in range(opt.n_input_samples):
        input_dw_path = os.path.join(opt.dataroot, 'DW', 'input{}'.format(i + 1), '{}.tif'.format(opt.test_id))
        input_s1_path = os.path.join(opt.dataroot, 'S1', 'input{}'.format(i + 1), '{}.tif'.format(opt.test_id))
        input_mask_path = os.path.join(opt.dataroot, 'Mask', 'input{}'.format(i + 1), '{}.tif'.format(opt.test_id))
        input_dw.append(read_tif(input_dw_path))
        input_s1.append(read_tif(input_s1_path))
        input_masks.append(read_tif(input_mask_path).astype(np.uint8))

    sample = {'input': {'DW': list(input_dw),
                        'S1': list(input_s1),
                        'masks': list(input_masks),
                        'coverage': [np.mean(mask) for mask in input_masks]
                        },
              'target': {'DW': [target_dw],
                         'S1': [target_s1],
                         },
              'patch_id': opt.test_id
              }

    A_DW, A_DW_mask = [], []
    A_S1 = []
    for i in range(opt.n_input_samples):
        A_DW_01 = sample['input']['DW'][i]
        A_DW.append(torch.unsqueeze(torch.tensor(A_DW_01), 0))
        A_DW_mask.append(torch.unsqueeze(torch.tensor(sample['input']['masks'][i].reshape((1, 256, 256))), 0))
        A_S1_01 = sample['input']['S1'][i]
        A_S1.append(torch.unsqueeze(torch.tensor(A_S1_01), 0))  # no need to rescale, keep at [0,2]
    B = torch.unsqueeze(torch.tensor(sample['target']['DW'][0]), 0)
    B_S1 = torch.unsqueeze(torch.tensor(sample['target']['S1'][0]), 0)

    return {'A_S1': A_S1, 'A_DW': A_DW, 'A_mask': A_DW_mask, 'B': B, 'B_S1': B_S1, 'patch_id': opt.test_id}


def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)

    # set gap pixels to yellow, other pixels to a blue color map
    for i in range(0, rows):
        for j in range(0, cols):
            index = int(gray_array[i, j] * 255)
            if index > 255:
                color_array[i, j] = (255, 255, 153)
            else:
                color_array[i, j] = color_map[index]

    return color_array


def get_color_map(size):
    colormap_int = np.zeros((256, 3), np.uint8)
    for i in range(0, size):
        colormap_int[i, 0] = np.int_(np.round(cm.Blues(i)[0] * 255.0))
        colormap_int[i, 1] = np.int_(np.round(cm.Blues(i)[1] * 255.0))
        colormap_int[i, 2] = np.int_(np.round(cm.Blues(i)[2] * 255.0))

    return colormap_int


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    opt.test_id = 3501
    opt.batch_size = 1
    opt.serial_batches = True       # disable data shuffling
    opt.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1             # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)   # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers

    if opt.eval:
        model.eval()  # test with eval mode. This affects layers like batchnorm and dropout.

    print('ID:', opt.test_id)

    data = get_data(opt)

    model.set_input(data)  # unpack data from data loader
    model.test()           # run inference

    # if opt.use_aux and not opt.no_stage2:
    #     fake = model.fake_B[0]
    # else:
    #     fake = model.fake_B

    metric = compute_metric(model.real_B, model.fake_B[0])
    print(metric)

    # output
    color_map = get_color_map(256)
    out_dir = os.path.join(opt.results_dir, opt.name, 'analysis')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    t_dw_path = os.path.join(out_dir, 'id{}_dw_target.tif'.format(opt.test_id))
    tif.imwrite(t_dw_path, gray2color(data['B'][0][0].numpy(), color_map))
    t_vv_path = os.path.join(out_dir, 'id{}_vv_target.tif'.format(opt.test_id))
    tif.imwrite(t_vv_path, data['B_S1'][0][0].numpy())
    t_vh_path = os.path.join(out_dir, 'id{}_vh_target.tif'.format(opt.test_id))
    tif.imwrite(t_vh_path, data['B_S1'][0][1].numpy())

    fake_path = os.path.join(out_dir, 'id{}_fake.tif'.format(opt.test_id))
    fake_color = gray2color(model.fake_B[0][0][0].cpu().numpy(), color_map)
    tif.imwrite(fake_path, fake_color)
    coarse_path = os.path.join(out_dir, 'id{}_coarse.tif'.format(opt.test_id))
    coarse_color = gray2color(model.fake_B[1][0][0].cpu().numpy(), color_map)
    tif.imwrite(coarse_path, coarse_color)

    for i in range(opt.n_input_samples):
        in_dw_path = os.path.join(out_dir, 'id{}_dw{}.tif'.format(opt.test_id, i))
        in_dw_data = data['A_DW'][i][0][0].numpy()
        tif.imwrite(in_dw_path, gray2color(in_dw_data, color_map))
        in_vh_path = os.path.join(out_dir, 'id{}_vv{}.tif'.format(opt.test_id, i))
        in_vh_data = data['A_S1'][i][0][0].numpy()
        tif.imwrite(in_vh_path, in_vh_data)
        in_vv_path = os.path.join(out_dir, 'id{}_vh{}.tif'.format(opt.test_id, i))
        in_vv_data = data['A_S1'][i][0][1].numpy()
        tif.imwrite(in_vv_path, in_vv_data)
        in_mask_path = os.path.join(out_dir, 'id{}_mask{}.tif'.format(opt.test_id, i))
        in_mask_data = data['A_mask'][i][0][0].numpy() * 255
        tif.imwrite(in_mask_path, in_mask_data)

    # least
    least_cloudy_idx = np.argsort([torch.sum(model.A_mask[t]).cpu().numpy() for t in range(opt.n_input_samples)])[0]
    least = data['A_DW'][least_cloudy_idx][0][0].numpy()
    least_path = os.path.join(out_dir, 'id{}_least.tif'.format(opt.test_id))
    tif.imwrite(least_path, gray2color(least, color_map))

    # mosaic
    mosaic_list = []
    for k in range(opt.n_input_samples):
        masked_t = np.where(model.A_mask[k].cpu().numpy(), np.nan, model.DW_input[k].cpu().numpy())
        mosaic_list.append(masked_t)
    # apply mean mosaicing
    mosaic = torch.tensor(np.nanmean(np.array(mosaic_list), 0))
    # for pixels that are nan across all time points: take neutral value
    mosaic[torch.isnan(mosaic)] = 2
    mosaic_path = os.path.join(out_dir, 'id{}_mosaic.tif'.format(opt.test_id))
    tif.imwrite(mosaic_path, gray2color(mosaic[0][0].numpy(), color_map))