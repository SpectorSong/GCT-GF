#  clip samples and save as a new structure
import os
import rasterio
from rasterio.windows import Window


def image_clip(in_path, new_dir, size, rate):
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)

    with rasterio.open(in_path) as dataset:
        # 获取影像的宽度和高度
        width = dataset.width
        height = dataset.height
        band = dataset.count
        meta = dataset.meta.copy()
        meta['width'], meta['height'] = size, size

        width_div = (width - size * rate) / (size * (1 - rate))
        height_div = (height - size * rate) / (size * (1 - rate))

        slices = []
        for i in range(int(height_div)):
            for j in range(int(width_div)):
                window = Window(int(i * size * (1 - rate)), int(j * size * (1 - rate)), size, size)
                slices.append(window)
        if width_div != int(width_div):  # 向前裁剪最后一列
            for i in range(int(height_div)):
                window = Window(int(i * size * (1 - rate)), width - size + 1, size, size)
                slices.append(window)
        if height_div != int(height_div):  # 向前裁剪最后一行
            for j in range(int(width_div)):
                window = Window(height - size + 1, int(j * size * (1 - rate)), size, size)
                slices.append(window)
        if height_div != int(height_div) and width_div != int(width_div):  # 裁剪右下角
            window = Window(height - size + 1, width - size + 1, size, size)
            slices.append(window)

        name_num = 1
        for window in slices:
            data = dataset.read(window=window)

            transform = dataset.window_transform(window)
            meta['transform'] = transform

            with rasterio.open(os.path.join(new_dir, str(name_num) + '.tif'), 'w', **meta) as dst:
                dst.write(data)

            name_num = name_num + 1


if __name__ == '__main__':
    in_dir = r'G:\reconstruction\data\Samples\DW_S1_samples'
    out_dir = r'G:\reconstruction\data\Samples\DW_S1_dataset'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #  裁剪图片,重复率为RepetitionRate
    crop_size = 256
    repetition_rate = 0

    RoIs = ['RoIs' + str(n) for n in range(1, 21)]
    in_modal = ['Dynamic World', 'Sentinel-1']
    out_modal = ['DW', 'S1']

    for time in range(1, 37):
        if time < 4:
            continue
        for roi in range(20):
            for modal in range(2):
                in_tif_path = os.path.join(in_dir, in_modal[modal], str(time), str(roi + 1) + '.tif')
                out_tif_dir = os.path.join(out_dir, RoIs[roi], out_modal[modal], str(time))
                image_clip(in_tif_path, out_tif_dir, crop_size, repetition_rate)
                print('Finish clipping RoI:{}, Modal:{}, Time Points:{}'.format(roi+1, out_modal[modal], time))
