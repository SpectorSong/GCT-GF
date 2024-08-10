import os
import numpy as np
import tifffile as tif
import rasterio
from matplotlib import cm


def gray2color(gray_array, color_map):
    rows, cols = gray_array.shape
    color_array = np.zeros((rows, cols, 3), np.uint8)
    # gray_array = np.clip(gray_array, 0, 1)

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
    # # name_list = ['sen12mscrts', 'stgan_4b', 'dsen2cr_4b', 'glf_cr', 'hs2p']
    # name_list = ['wpgct_nostage2_nogan', 'wpgct_nostage1_nogan', 'wpgct_nostage2', 'wpgct_nostage1', 'wpgct_nogan']
    # # name_list = ['wpgct_mono', 'wpgct_nos1', 'wpgct_nodw']
    # patch_id = '3501'
    #
    # for name in name_list:
    #     in_path = os.path.join(r'G:\reconstruction\test', name, r'test_30_all\images', patch_id+'_model_output.tif')
    #     out_path = os.path.join(r'G:\reconstruction\article\ablation\model_images', name+'_'+patch_id+'_fake.tif')
    #
    #     if os.path.exists(in_path):
    #         in_tif = rasterio.open(in_path)
    #         in_data = in_tif.read().astype(np.float32)
    #
    #         color_map = get_color_map(256)
    #         out_data = gray2color(in_data[0], color_map)
    #         tif.imwrite(out_path, out_data)
    in_path = r'G:\reconstruction\large_range\input\qinghaihu\B_DW_2023-06-21.tif'
    out_path = r'G:\reconstruction\large_range\input\qinghaihu\B_DW_2023-06-21_color.tif'
    in_tif = rasterio.open(in_path)
    in_data = in_tif.read().astype(np.float32)
    in_data = np.where(np.isnan(in_data), 2, in_data)
    color_map = get_color_map(256)
    out_data = gray2color(in_data[0], color_map)
    tif.imwrite(out_path, out_data)
