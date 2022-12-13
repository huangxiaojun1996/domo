import os
import os.path
import sys
import torch
import torch.utils.data as data
import cv2
import numpy as np
import re
import json
from data.data_augment import preproc
from data.config import cfg_mnet


class WiderFaceDetection(data.Dataset):
    def __init__(self, txt_path, preproc=None):
        self.preproc = preproc
        self.imgs_path = []
        self.words = []
        f = open(txt_path, 'r')
        lines = f.readlines()
        isFirst = True
        labels = []
        for line in lines:
            line = line.rstrip()
            if line.startswith('#'):
                if isFirst is True:
                    isFirst = False
                else:
                    labels_copy = labels.copy()
                    self.words.append(labels_copy)
                    labels.clear()
                path = line[2:]
                path = txt_path.replace('label.txt', 'images/') + path
                self.imgs_path.append(path)
            else:
                line = line.split(' ')
                label = [float(x) for x in line]
                labels.append(label)

        self.words.append(labels)

    def __len__(self):
        return len(self.imgs_path)

    def __getitem__(self, index):
        img = cv2.imread(self.imgs_path[index])
        height, width, _ = img.shape

        labels = self.words[index]
        annotations = np.zeros((0, 13))
        if len(labels) == 0:
            return annotations
        for idx, label in enumerate(labels):
            annotation = np.zeros((1, 13))
            #   x y w h x1 y1 0.0 x2 y2 0.0 x3 y3 0.0 x4 y4 0.0
            # bbox
            annotation[0, 0] = label[0]  # x1
            annotation[0, 1] = label[1]  # y1
            annotation[0, 2] = label[0] + label[2]  # x2
            annotation[0, 3] = label[1] + label[3]  # y2

            # landmarks
            annotation[0, 4] = label[4]  # l0_x
            annotation[0, 5] = label[5]  # l0_y
            annotation[0, 6] = label[7]  # l1_x
            annotation[0, 7] = label[8]  # l1_y
            annotation[0, 8] = label[10]  # l2_x
            annotation[0, 9] = label[11]  # l2_y
            annotation[0, 10] = label[13]  # l3_x
            annotation[0, 11] = label[14]  # l3_y

            if (annotation[0, 4] < 0):
                annotation[0, 12] = -1
            else:
                annotation[0, 12] = 1

            annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


def detection_collate(batch):
    """Custom collate fn for dealing with batches of images that have a different
    number of associated object annotations (bounding boxes).

    Arguments:
        batch: (tuple) A tuple of tensor images and lists of annotations

    Return:
        A tuple containing:
            1) (tensor) batch of images stacked on their 0 dim
            2) (list of tensors) annotations for a given image are stacked on 0 dim
    """
    targets = []
    imgs = []
    for _, sample in enumerate(batch):
        for _, tup in enumerate(sample):
            if torch.is_tensor(tup):
                imgs.append(tup)
            elif isinstance(tup, type(np.empty(0))):
                annos = torch.from_numpy(tup).float()
                targets.append(annos)

    return (torch.stack(imgs, 0), targets)


class myDataset(data.Dataset):

    def __init__(self, image_dir, label_dir, preproc=None):
        self.preproc = preproc
        self.image_list = []
        self.label_list = []
        names = os.listdir(image_dir)
        for name in names:
            self.image_list.append(os.path.join(image_dir, name))
            self.label_list.append(os.path.join(label_dir, "{}.json".format(name.split(".")[0])))

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img = cv2.imread(self.image_list[index])
        height, width, _ = img.shape
        # 构造label标签 x0,y0,x3,y3,x0,y0,x1,y1,x2,y2,x3,y3
        annotations = np.zeros((0, 13))
        with open(self.label_list[index], "r") as f:
            label = ""
            for line in f:
                label += re.sub("\n| ", "", line)
        label = np.array(json.loads(label)["shapes"][0]["points"])
        annotation = np.zeros((1, 13))
        annotation[0, 0] = label[0][0]  # x1
        annotation[0, 1] = label[0][1]  # y1
        annotation[0, 2] = label[2][0]
        annotation[0, 3] = label[2][1]

        # landmarks
        annotation[0, 4] = label[0][0]  # l0_x
        annotation[0, 5] = label[0][1]  # l0_y
        annotation[0, 6] = label[1][0]  # l1_x
        annotation[0, 7] = label[1][1]  # l1_y
        annotation[0, 8] = label[2][0]  # l2_x
        annotation[0, 9] = label[2][1]  # l2_y
        annotation[0, 10] = label[3][0]  # l3_x
        annotation[0, 11] = label[3][1]  # l3_y
        if (annotation[0, 4] < 0):
            annotation[0, 12] = -1
        else:
            annotation[0, 12] = 1
        annotations = np.append(annotations, annotation, axis=0)
        target = np.array(annotations)
        if self.preproc is not None:
            img, target = self.preproc(img, target)

        return torch.from_numpy(img), target


if __name__ == '__main__':
    cfg = cfg_mnet
    rgb_mean = (104, 117, 123)
    img_dim = cfg['image_size']
    dataset = myDataset(
        image_dir="D:/PYprogram/TensorFlow/picture/33s",
        label_dir="D:/PYprogram/TensorFlow/picture/33_labelme",
        preproc=preproc(img_dim, rgb_mean))

