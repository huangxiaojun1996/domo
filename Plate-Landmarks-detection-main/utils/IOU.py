#!/usr/bin/env python
# -*- coding: UTF-8 -*-
'''
@Project ：Plate-Landmarks-detection-main 
@File    ：IOU.py
@Author  ：huangxj
@Date    ：2022/12/12 19:17 
'''

import numpy as np
from shapely.geometry import Polygon


def get_ious(point_a, point_b):
    def get_iou(a: np.ndarray, b: np.ndarray):
        a = a.reshape(4, 2)
        b = b.reshape(4, 2)
        poly_a = Polygon(a).convex_hull
        poly_b = Polygon(b).convex_hull
        if poly_a.intersects(poly_b):
            inner_area = poly_a.intersection(poly_b).area
            all_area = poly_a.area + poly_b.area - inner_area
            iou = inner_area / all_area  # 交集/并集
        else:
            iou = 0
        return iou

    num = 0
    for i in range(len(point_a)):
        iou = get_iou(point_a[i], point_b[i])
        num += iou
    return num


if __name__ == '__main__':
    a = [[2, 0, 2, 2, 0, 0, 0, 2], [1, 1, 4, 1, 4, 4, 1, 4]]
    a = np.array(a)#.reshape(2, 4, 2)
    b = [[1, 1, 4, 1, 4, 4, 1, 4], [2, 0, 2, 2, 0, 0, 0, 2]]
    b = np.array(b)#.reshape(2, 4, 2)
    print(get_ious(a, b))