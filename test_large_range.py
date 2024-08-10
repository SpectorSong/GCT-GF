import time
import os
import numpy as np
import rasterio
from rasterio.windows import Window
import tifffile as tif
import torch

from options.test_options import TestOptions
from models import create_model


def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img = (img - oldMin) / oldRange
    return img


def process_SAR(img):
    # project SAR to [0, 1] range
    dB_min, dB_max = [-25.0, -32.5], [0, 0]
    img_new = np.array([rescale(img[0], dB_min[0], dB_max[0]), rescale(img[1], dB_min[1], dB_max[1])])
    return img_new


def tif_predict(model, in_dir, out_dir, batch, size=256, rate=0.2):
    s_time = time.time()

    # 裁剪窗口
    tif_list = os.listdir(in_dir)
    B_DW_path = ''
    for tif in tif_list:
        if 'B_DW' in tif:
            B_DW_path = os.path.join(in_dir, tif)
    with rasterio.open(B_DW_path) as dataset:
        # 获取影像的宽度和高度
        width = dataset.width
        height = dataset.height
        meta = dataset.meta.copy()

        width_div = width / (size * (1 - rate))
        height_div = height / (size * (1 - rate))

        slices = []
        for i in range(int(height_div)):
            for j in range(int(width_div)):
                window = Window(int(i * size * (1 - rate)), int(j * size * (1 - rate)), size, size)
                slices.append(window)
        if width_div != int(width_div):  # 向前裁剪最后一列
            for i in range(int(height_div)):
                window = Window(int(i * size * (1 - rate)), width - size, size, size)
                slices.append(window)
        if height_div != int(height_div):  # 向前裁剪最后一行
            for j in range(int(width_div)):
                window = Window(height - size, int(j * size * (1 - rate)), size, size)
                slices.append(window)
        if height_div != int(height_div) and width_div != int(width_div):  # 裁剪右下角
            window = Window(height - size, width - size, size, size)
            slices.append(window)

    # slices_batch = [slices[i:i+batch] for i in range(0, len(slices), batch)]

    # 设置模型输入输出
    dst = rasterio.open(os.path.join(out_dir, 'gap_filling_' + B_DW_path.split('_')[-1]), 'w', **meta)
    # for slices in slices_batch:
    for id, window in enumerate(slices):
        if id % 20 == 0:
            now_time = time.time()
            print('Predicting %04d-th slice, time cost: %.2f s' % (id+1, now_time-s_time), flush=True)

        A_DW_list, A_S1_list, A_mask_list = [], [], []
        for tif in tif_list:
            with rasterio.open(os.path.join(in_dir, tif)) as dataset:
                if 'A_S1' in tif:
                    A_S1 = dataset.read(window=window)
                    A_S1 = np.where(np.isnan(A_S1), 1, process_SAR(A_S1))
                    A_S1_list.append(torch.tensor(A_S1, dtype=torch.float32).unsqueeze(0))
                elif 'A_DW' in tif:
                    A_DW = dataset.read(window=window)
                    A_DW = np.where(np.isnan(A_DW), 2, A_DW)
                    A_mask = np.where(A_DW == 2, 1, 0).astype(np.uint8)
                    A_DW_list.append(torch.tensor(A_DW, dtype=torch.float32).unsqueeze(0))
                    A_mask_list.append(torch.tensor(A_mask).unsqueeze(0))
                elif 'B_DW' in tif:
                    B_DW = dataset.read(window=window)
                    B_DW = np.where(np.isnan(B_DW), 2, B_DW)
                    B_DW = torch.tensor(B_DW, dtype=torch.float32).unsqueeze(0)
                elif 'B_S1' in tif:
                    B_S1 = dataset.read(window=window)
                    B_S1 = np.where(np.isnan(B_S1), 1, process_SAR(B_S1))
                    B_S1 = torch.tensor(B_S1, dtype=torch.float32).unsqueeze(0)

        if np.nanmax(B_DW) < 2:
            dst.write(B_DW.cpu().numpy().squeeze(0), window=window)
        else:
            data = {'A_S1': A_S1_list, 'A_DW': A_DW_list, 'A_mask': A_mask_list, 'B': B_DW, 'B_S1': B_S1, 'patch_id': id + 1}
            model.set_input(data)
            model.test()
            fake = model.fake_B[0].cpu().numpy()
            filled = np.where(B_DW == 2, fake, B_DW)
            filled = np.clip(filled, 0., 1.)
            dst.write(filled.squeeze(0), window=window)

            # visual_dict = {'input1_DW': model.DW_input[0][0].cpu().numpy(),
            #                'input2_DW': model.DW_input[1][0].cpu().numpy(),
            #                'input3_DW': model.DW_input[2][0].cpu().numpy(),
            #                'model_output': fake}
            # save_images(visual_dict, out_dir, id + 1)

    dst = None


def save_images(visuals, image_dir, patch):
    image_dir = os.path.join(image_dir, 'images')
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for label, im_data in visuals.items():
        image_name = '%s_%s.tif' % (patch, label)
        img_path = os.path.join(image_dir, image_name)
        tif.imwrite(img_path, im_data)


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.serial_batches = True       # disable data shuffling
    opt.no_flip = True              # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1             # no visdom display; the test code saves the results to a HTML file.

    model = create_model(opt)       # create a model given opt.model and other options
    model.setup(opt)                # regular setup: load and print networks; create schedulers
    if opt.eval:
        model.eval()

    input_dir = r'G:\reconstruction\large_range\input\qinghaihu'
    output_dir = r'G:\reconstruction\large_range\reconstruction\qinghaihu'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    tif_predict(model, input_dir, output_dir, opt.batch_size, rate=0.2)

