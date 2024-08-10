import numpy as np
from osgeo import gdal


def readTif(img_path):
    bands_order = [1]     # R,G,B
    band_num = len(bands_order)
    dataset = gdal.Open(img_path, gdal.GA_ReadOnly)
    cols = dataset.RasterXSize
    rows = dataset.RasterYSize

    data = np.empty([rows, cols, band_num])
    for k in range(band_num):
        band = dataset.GetRasterBand(bands_order[k])
        oneband_data = np.array(band.ReadAsArray())
        data[:, :, k] = oneband_data

    del dataset
    return data


def compute_metric(real_path, fake_path):
    real_data = readTif(real_path)
    fake_data = readTif(fake_path)
    real = real_data[real_data < 2]
    fake = fake_data[real_data < 2]

    rmse = np.sqrt(np.mean(np.square(real - fake)))
    psnr = 20 * np.log10(1 / rmse)
    mae = np.mean(np.abs(real - fake))

    mask1 = real.astype(bool)
    mask2 = fake.astype(bool)
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    intersection_pixels = np.sum(intersection)
    union_pixels = np.sum(union)
    iou = intersection_pixels / union_pixels if union_pixels > 0 else 0

    return {'RMSE': rmse.item(),
            'MAE': mae.item(),
            'PSNR': psnr.item(),
            'IoU': iou.item()}


if __name__ == '__main__':
    real_path = r'E:\nature-oriented\half_month\51_2022-07-16_70_30\51_image.tif'
    fake_path = r'E:\nature-oriented\half_month\51_2022-07-16_70_30\51_fillwater.tif'

    metrics = compute_metric(real_path, fake_path)
    print(metrics)
