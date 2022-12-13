#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Plate-Landmarks-detection-main 
@File    ：model_prediction.py
@Author  ：huangxj
@Date    ：2022/11/29 10:43
使用1.2版本无法读取1.6以上版本保存的模型
'''
import re
import os
import json
import time
import argparse
import torch
from data import config
import numpy as np
import cv2
from models.retinaface import RetinaFace
import torch.backends.cudnn as cudnn
from detect import check_keys, remove_prefix, load_model
from layers.functions.prior_box import PriorBox
from utils.box_utils import decode, decode_landm
from utils.nms.py_cpu_nms import py_cpu_nms
from shapely.geometry import Polygon
from utils.IOU import get_ious

parser = argparse.ArgumentParser(description='Retinaface')
parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')
parser.add_argument('--cpu', action="store_true", default=True, help='Use cpu inference')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('--save_model', action="store_true", default=False, help='save full model')
# parser.add_argument('--input', default='./15.jpg', help='image input')
parser.add_argument('--vis_thres', default=0.6, type=float, help='visualization_threshold')
parser.add_argument('--save_image', action="store_true", default=True, help="save full image")

torch.set_grad_enabled(False)
device = torch.device("cpu")
resize = 1


class Pred():

    def __init__(self):
        self.args = parser.parse_args()
        self._config()  # 初始化模型参数
        self._net()  # 初始化模型

    def _config(self):
        if self.args.network == "mobile0.25":
            self.cfg = config.cfg_mnet
        elif self.args.network == "resnet50":
            self.cfg = config.cfg_re50

    def _net(self):
        """
        导入模型，模型路径存放在args中
        Returns:
        """
        net = RetinaFace(cfg=self.cfg, phase='test')
        net = load_model(net, self.args.trained_model, self.args.cpu)
        net.eval()
        print('Finished loading model!')
        cudnn.benchmark = True
        self.net = net.to(device)

    def _image(self, path):
        """
        对获取的图片进行初步处理
        Args:path: 图片路径
        Returns:原始图像，修改后图像
        """
        img_raw = cv2.imread(path, cv2.IMREAD_COLOR)
        img = np.float32(img_raw)
        im_height, im_width, _ = img.shape
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(device)
        scale = scale.to(device)
        return img_raw, img, scale, im_height, im_width

    def _pred(self, img):
        """
        模型预测
        Args:img:
        Returns:
        """
        tic = time.time()
        loc, conf, landms = self.net(img)  # forward pass
        print('net forward time: {:.4f}'.format(time.time() - tic))
        return loc, conf, landms

    def handle_pred(self, im_height, im_width, img, scale, loc, conf, landms):
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2]])
        scale1 = scale1.to(device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.args.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1][:self.args.top_k]  # 从小到大排序，并返回相应序列元素的下标，再进行转置。
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)  # 将2个数组按水平方向组合
        keep = py_cpu_nms(dets, self.args.nms_threshold)
        # keep = nms(dets, args.nms_threshold,force_cpu=args.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        dets = dets[:self.args.keep_top_k, :]
        landms = landms[:self.args.keep_top_k, :]

        dets = np.concatenate((dets, landms), axis=1)
        return dets

    def main(self, path):
        img_raw, img, scale, im_height, im_width = self._image(path=path)
        loc, conf, landms = self._pred(img=img)
        dets = self.handle_pred(im_height, im_width, img, scale, loc, conf, landms)

        for det in dets:
            p1 = (int(det[5]), int(det[6]))
            p2 = (int(det[7]), int(det[8]))
            p3 = (int(det[9]), int(det[10]))
            p4 = (int(det[11]), int(det[12]))

            cv2.line(img=img_raw, pt1=p1, pt2=p2, color=(0, 0, 255), thickness=3)
            cv2.line(img=img_raw, pt1=p2, pt2=p3, color=(0, 0, 255), thickness=3)
            cv2.line(img=img_raw, pt1=p3, pt2=p4, color=(0, 0, 255), thickness=3)
            cv2.line(img=img_raw, pt1=p4, pt2=p1, color=(0, 0, 255), thickness=3)
            cv2.imwrite('./res/{}'.format(path.split("\\")[-1]), img_raw)
        """
        # 删除注释
        cv2.imshow("image", img_raw)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """
        if dets.shape[0] == 0:
            return np.array([[0, 0, 0, 0, 0, 0, 0, 0]])
        else:
            return np.array([dets[0, -8:]])


def Validation_Set_IOU(root, dir):
    """
    Args:
        root: 图片路径
        dir: 标签路径
    Returns: 返回验证集的iou
    """

    def label(path):
        with open(path, "r") as f:
            label = ""
            for line in f:
                label += re.sub("\n| ", "", line)
        return np.array(json.loads(label)["shapes"][0]["points"]).reshape(1, 8)

    model = Pred()
    names = os.listdir(root)
    ious = 0
    for name in names:
        name = name.split(".")[0]
        pred = model.main(path=os.path.join(root, "{}.jpg".format(name)))
        targ = label(path=os.path.join(dir, "{}.json".format(name)))
        ious += get_ious(pred, targ)

    return ious / len(names)


if __name__ == '__main__':
    root = "D:\\PYprogram\\TensorFlow\\picture\\38s"
    dir = "D:\\PYprogram\\TensorFlow\\picture\\38_labelme"
    iou = Validation_Set_IOU(root, dir)
    print(iou)
