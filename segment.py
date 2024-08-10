import os
import numpy as np
import tifffile as tif
import rasterio


def thresh_seg(in_path, thresh):
    in_data = tif.imread(in_path)
    if len(in_data.shape) > 2:
        in_data = in_data.squeeze(0)
    seg_data = np.where(in_data > thresh, 1, 0).reshape(in_data.shape).astype(np.uint8)

    return seg_data


if __name__ == '__main__':
    seg_thresh = 0.6
    in_path = r'G:\reconstruction\large_range\reconstruction\qinghaihu\gap_filling_2023-06-21.tif'
    seg_color_path = in_path.split('.tif')[0] + '_seg{:.1f}_color.tif'.format(seg_thresh)

    seg_img = thresh_seg(in_path, thresh=seg_thresh)

    band_r = np.where(seg_img > 0, 36, 240).reshape(seg_img.shape)
    band_g = np.where(seg_img > 0, 116, 245).reshape(seg_img.shape)
    band_b = np.where(seg_img > 0, 183, 253).reshape(seg_img.shape)
    color_array = np.array([band_r, band_g, band_b]).astype(np.uint8)
    # color_array = np.transpose(color_array, (1, 2, 0)).astype(np.uint8)

    with rasterio.open(in_path) as dataset:
        meta = dataset.meta.copy()
        meta['dtype'] = np.uint8
        meta['count'] = 3
    dst = rasterio.open(seg_color_path, 'w', **meta)
    dst.write(color_array)
    # tif.imwrite(seg_color_path, color_array)
