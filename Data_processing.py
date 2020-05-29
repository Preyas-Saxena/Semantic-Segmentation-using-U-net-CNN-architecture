import os
import numpy as np
from skimage import io
from skimage.transform import resize
from skimage.color import rgb2gray

voc_root_folder='C:/Users/Preyas/Desktop/Lectures/Lec_2nd_sem/CV/Project/VOCdevkit'
image_size=128

def build_seg_onemask_dataset(filename):
    txt_fname = '%s/VOC2009/ImageSets/Segmentation/%s' % (voc_root_folder, filename)
    with open(txt_fname, 'r') as f:
        images = f.read().split()

    filter = ['person']
    annotation_folder = os.path.join(voc_root_folder, "VOC2009/Annotations/")
    annotation_files = os.listdir(annotation_folder)
    filtered_filenames = []

    for a_f in annotation_files:
        tree = etree.parse(os.path.join(annotation_folder, a_f))
        if np.any([tag.text == filt for tag in tree.iterfind(".//name") for filt in filter]):
            filtered_filenames.append(a_f[:-4])

    anno_set= set(filtered_filenames)
    seg_set= set(images)
    common_set= anno_set.intersection(seg_set)
    common_list= list(common_set)

    features, labels = [None] * len(common_list), [None] * len(common_list)

    for i, fname in enumerate(common_list):
        features[i] = resize(io.imread('%s/VOC2009/JPEGImages/%s.jpg' % (voc_root_folder, fname)), (image_size, image_size, 3))
        seg = cv2.imread('%s/VOC2009/SegmentationClass/%s.png' % (voc_root_folder, fname))
        seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
        seg = resize(seg, (image_size, image_size, 3))
        lower = (192 / 255, 128 / 255, 128 / 255)
        higher = (192 / 255, 128 / 255, 128 / 255)
        mask = cv2.inRange(seg, lower, higher)
        label = np.where(mask != 0, 1, mask)
        label = np.expand_dims(label, axis=2)
        labels[i] = label

    feat_array = np.array(features)
    label_array = np.array(labels)
    return feat_array, label_array
