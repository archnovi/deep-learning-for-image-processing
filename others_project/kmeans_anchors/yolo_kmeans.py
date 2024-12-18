import numpy as np


def wh_iou(wh1, wh2):
    # Returns the nxm IoU matrix. wh1 is nx2, wh2 is mx2
    # 加维度以便应用广播机制
    wh1 = wh1[:, None]  # [N,1,2]
    wh2 = wh2[None]  # [1,M,2]
    inter = np.minimum(wh1, wh2).prod(2)  # [N,M]  # 注意：这里是假设所有的gtbox的左上角是对齐的，才能这么计算
    return inter / (wh1.prod(2) + wh2.prod(2) - inter)  # iou = inter / (area1 + area2 - inter)


def k_means(boxes, k, dist=np.median):
    """
    yolo k-means methods
    refer: https://github.com/qqwweee/keras-yolo3/blob/master/kmeans.py
    Args:
        boxes: 需要聚类的bboxes
        k: 簇数(聚成几类)
        dist: 更新簇坐标的方法(默认使用中位数，比均值效果略好)
    """
    box_number = boxes.shape[0]
    last_nearest = np.zeros((box_number,))
    # np.random.seed(0)  # 固定随机数种子

    # init k clusters
    # 随机算则k个簇心，replace设置成False时，k个簇心是没有重复的
    clusters = boxes[np.random.choice(box_number, k, replace=False)]

    while True:
        distances = 1 - wh_iou(boxes, clusters)  # 用1-iou表示box到簇心的距离，这比欧氏距离效果更好 假设k=9，distances shape: box个数，9
        current_nearest = np.argmin(distances, axis=1) # 表示每个box寻找离他最近的簇心，得到【1,8,8,6,6，4，.....】 表示每个box离他最近的簇心索引
        if (last_nearest == current_nearest).all(): # 不断循环，如果最后一次计算和上一次计算的结果相同，表示分类结束了
            break  # clusters won't change
        for cluster in range(k): # 遍历每个簇心
            # update clusters
            clusters[cluster] = dist(boxes[current_nearest == cluster], axis=0) # 取出在这个簇的box，用dist方法重新计算簇心

        last_nearest = current_nearest

    return clusters
