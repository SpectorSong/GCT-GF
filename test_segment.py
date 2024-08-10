import os
import numpy as np
import tifffile as tif


def thresh_seg(in_path, out_dir, id, thresh):
    in_data = tif.imread(in_path)
    if len(in_data.shape) > 2:
        in_data = in_data.squeeze(0)
    seg_data = np.where(in_data > thresh, 1, 0).reshape(in_data.shape).astype(np.uint8)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    if 'output' in in_path:
        out_path = os.path.join(out_dir, '{}_output_seg.tif'.format(id))
    else:
        out_path = os.path.join(out_dir, '{}_target_seg.tif'.format(id))
    tif.imwrite(out_path, seg_data * 255)

    return seg_data


def segment_metric(real, pred):
    label = 2 * real + pred
    count = np.bincount(label.reshape(-1), minlength=2**2)
    confusion_matrix = count.reshape(2, 2)

    precision = np.diag(confusion_matrix) / confusion_matrix.sum(axis=0)

    intersection = np.diag(confusion_matrix)
    union = np.sum(confusion_matrix, axis=1) + np.sum(confusion_matrix, axis=0) - np.diag(confusion_matrix)
    iou = intersection / union

    recall = np.diag(confusion_matrix) / confusion_matrix.sum(axis=1)

    return {'Precision': precision[1],
            'IoU': iou[1],
            'Recall': recall[1]}


def save_metric(metric, patch_list, path, thresh):
    f = open(path + f'/seg_{thresh}_metric.txt', 'w')

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


if __name__ == '__main__':
    target_dir = r'G:\reconstruction\data\Samples\DW_S1_dataset_nearest_input_new\DW\target'
    model_output_dir = r'G:\reconstruction\test\sen12mscrts\test_latest_100'
    seg_thresh = 0.5
    segment_dir = model_output_dir + '_seg_{}'.format(seg_thresh)

    output_list = os.listdir(os.path.join(model_output_dir, 'images'))
    id_list = [name.split('_')[0] for name in output_list if 'output' in name]

    metric_list = []
    patch_list = []

    for id in id_list:
        print('Processing {}'.format(id))
        output_path = os.path.join(model_output_dir, 'images', '{}_model_output.tif'.format(id))
        target_path = os.path.join(target_dir, '{}.tif'.format(id))
        output_seg = thresh_seg(output_path, os.path.join(segment_dir, 'segments'), id, seg_thresh)
        target_seg = thresh_seg(target_path, os.path.join(segment_dir, 'segments'), id, seg_thresh)

        metric_list.append(segment_metric(target_seg, output_seg))
        patch_list.append(id)

    save_metric(metric_list, patch_list, segment_dir, seg_thresh)
