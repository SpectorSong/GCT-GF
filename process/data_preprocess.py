import os
from natsort import natsorted
import rasterio
import numpy as np
from tqdm import tqdm


def read_tif(path_IMG):
    tif = rasterio.open(path_IMG)
    return tif


def read_img(tif):
    return tif.read().astype(np.float32)


def rescale(img, oldMin, oldMax):
    oldRange = oldMax - oldMin
    img = (img - oldMin) / oldRange
    return img


def process_SAR(img):
    # project SAR to [0, 1] range
    dB_min, dB_max = [-25.0, -32.5], [0, 0]
    img_new = np.array([rescale(img[0], dB_min[0], dB_max[0]), rescale(img[1], dB_min[1], dB_max[1])])
    return img_new


def get_paths():  # assuming for the same ROI+num, the patch numbers are the same
    paths = []
    num_pdx = 0
    for roi in tqdm(ROI):
        if roi not in splits[split]:
            continue
        roi_path = os.path.join(root_dir, roi)
        path_dw_t, path_s1_t = [], []
        for tdx in time_points:
            # working with directory under time stamp tdx
            path_dw_complete = os.path.join(roi_path, modalities[0], str(tdx + 1))
            path_s1_complete = os.path.join(roi_path, modalities[1], str(tdx + 1))

            # same as complete paths, truncating root directory's path
            path_dw = os.path.join(roi, modalities[0], str(tdx + 1))
            path_s1 = os.path.join(roi, modalities[1], str(tdx + 1))

            # get list of files which contains all the patches at time tdx
            dw_t = natsorted([os.path.join(path_dw, f) for f in os.listdir(path_dw_complete) if
                              (os.path.isfile(os.path.join(path_dw_complete, f)) and ".tif" in f)])
            s1_t = natsorted([os.path.join(path_s1, f) for f in os.listdir(path_s1_complete) if
                              (os.path.isfile(os.path.join(path_s1_complete, f)) and ".tif" in f)])
            # same number of patches
            assert len(dw_t) == len(s1_t)

            # sort via file names according to patch number and store
            path_dw_t.append(dw_t)
            path_s1_t.append(s1_t)

        # for each patch of the ROI, collect its time points and make this one sample
        for pdx in range(len(path_s1_t[0])):
            sample = {"DW": [path_dw_t[tdx][pdx] for tdx in time_points],
                      "S1": [path_s1_t[tdx][pdx] for tdx in time_points]}

            paths.append(sample)
        num_pdx = num_pdx + len(path_s1_t[0])
    return paths, num_pdx


root_dir = 'G:\\reconstruction\\data\\Samples\\DW_S1_dataset'
new_root_dir = 'G:\\reconstruction\\data\\Samples\\DW_S1_dataset_nearest_input_new'
ROI = ['RoIs' + str(n) for n in range(1, 21)]
splits = {'all': ROI, 'test': ['RoIs4', 'RoIs6', 'RoIs12', 'RoIs14']}
splits['train'] = [r for r in splits['all'] if r not in splits['test']]
split = 'all'
modalities = ["DW", "S1"]
time_points = list(range(36))
paths, n_pdx = get_paths()
print('Total number of patches: {}'.format(n_pdx))

min_cov = 0.1
max_cov = 0.9
n_input_t = 3

modalities.append('Mask')
input_types = ['target', 'input1', 'input2', 'input3']
for m in modalities:
    for i in input_types:
        output_dir = os.path.join(new_root_dir, m, i)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

n_valid_patches = 0
for pdx in range(n_pdx):
    s1_tif = [read_tif(os.path.join(root_dir, img)) for img in paths[pdx]['S1']]
    dw_tif = [read_tif(os.path.join(root_dir, img)) for img in paths[pdx]['DW']]
    # coord = [list(tif.bounds) for tif in dw_tif]

    s1 = [process_SAR(read_img(img)) for img in s1_tif]
    dw = [read_img(img) for img in dw_tif]

    masks = [np.where(img == 2, 1, 0).astype(np.uint8) for img in dw]
    coverage = [np.mean(mask) for mask in masks]
    coverage_idx = [i for i in np.argsort(coverage) if coverage[i] == 0 and np.max(s1[i]) <= 1]

    inputs_idx = []
    target_idx = 0
    for cloudless_idx in coverage_idx:
        #  最临近的数据
        near_idx = np.argsort([abs(tdx - cloudless_idx) for tdx in range(len(coverage))])
        #  选取其中S1数据完整的，且不使用简单样本（没有缺失）或无信息的样本
        inputs_idx = [tdx for tdx in near_idx if np.max(s1[tdx]) <= 1 and tdx != cloudless_idx
                      and min_cov <= coverage[tdx] <= max_cov][:n_input_t]
        if len(inputs_idx) == n_input_t:
            target_idx = cloudless_idx
            break
    # ... if not exists then take the nearest n_input_t samples (except target patch)
    if len(inputs_idx) < n_input_t:
        for cloudless_idx in coverage_idx:
            #  最临近的数据
            near_idx = np.argsort([abs(tdx - cloudless_idx) for tdx in range(len(coverage))])
            #  放宽对DW数据完整性的限制
            inputs_idx = [tdx for tdx in near_idx if np.max(s1[tdx]) <= 1 and tdx != cloudless_idx
                          and 0 < coverage[tdx] < 1][:n_input_t]
            if len(inputs_idx) == n_input_t:
                target_idx = cloudless_idx
                break
    # 放宽限制后仍然不满足，则当前Patch不符合要求
    if len(inputs_idx) < n_input_t:
        print('Patch:{} is invalid.'.format(pdx))
        continue

    target_s1, target_dw, target_mask = np.array(s1)[target_idx], np.array(dw)[target_idx], \
                                        np.array(masks)[target_idx]
    input_s1, input_dw, input_masks = np.array(s1)[inputs_idx], np.array(dw)[inputs_idx], \
                                      np.array(masks)[inputs_idx]

    target_dw_meta = dw_tif[target_idx].meta.copy()
    target_s1_meta = s1_tif[target_idx].meta.copy()
    target_mask_meta = target_dw_meta.copy()
    target_mask_meta['dtype'] = 'int8'
    target_dw_path = os.path.join(new_root_dir, 'DW', 'target', '{}.tif'.format(n_valid_patches))
    with rasterio.open(target_dw_path, 'w',  **target_dw_meta) as dst:
        dst.write(target_dw)
    target_s1_path = os.path.join(new_root_dir, 'S1', 'target', '{}.tif'.format(n_valid_patches))
    with rasterio.open(target_s1_path, 'w',  **target_s1_meta) as dst:
        dst.write(target_s1)
    target_mask_path = os.path.join(new_root_dir, 'Mask', 'target', '{}.tif'.format(n_valid_patches))
    with rasterio.open(target_mask_path, 'w',  **target_mask_meta) as dst:
        dst.write(target_mask)

    for i in range(n_input_t):
        input_dw_meta = dw_tif[inputs_idx[i]].meta.copy()
        input_s1_meta = s1_tif[inputs_idx[i]].meta.copy()
        input_mask_meta = input_dw_meta.copy()
        input_mask_meta['dtype'] = 'int8'
        input_dw_path = os.path.join(new_root_dir, 'DW', 'input{}'.format(i+1), '{}.tif'.format(n_valid_patches))
        with rasterio.open(input_dw_path, 'w', **input_dw_meta) as dst:
            dst.write(input_dw[i])
        input_s1_path = os.path.join(new_root_dir, 'S1', 'input{}'.format(i+1), '{}.tif'.format(n_valid_patches))
        with rasterio.open(input_s1_path, 'w', **input_s1_meta) as dst:
            dst.write(input_s1[i])
        input_mask_path = os.path.join(new_root_dir, 'Mask', 'input{}'.format(i+1), '{}.tif'.format(n_valid_patches))
        with rasterio.open(input_mask_path, 'w', **input_mask_meta) as dst:
            dst.write(input_masks[i])

    n_valid_patches = n_valid_patches + 1
    print('Patch:{} is valid.'.format(pdx))

print("Number of valid patches: {}".format(n_valid_patches))
