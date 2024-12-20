import random
import numpy as np
from tqdm import tqdm
from scipy.cluster.vq import kmeans

from read_voc import VOCDataSet
from yolo_kmeans import k_means, wh_iou


def anchor_fitness(k: np.ndarray, wh: np.ndarray, thr: float):  # mutation fitness
    r = wh[:, None] / k[None] # 如果聚类k和标注框重合度更高，r越接近1。其他的会比1小也可能比1大
    x = np.minimum(r, 1. / r).min(2)  # ratio metric   np.minimum(r, 1. / r) 将比1大的情况都转为比1小。 min求（高度与高度的ratio，宽度与宽度的ratio）两个数的最小值（匹配最差的边）
    # x = wh_iou(wh, k)  # iou metric
    best = x.max(1) # best得到的是每个标注框与其匹配度最高的k（anchor）的匹配最差的边   max(1)表示第二个维度（也就是anchor）
    f = (best * (best > thr).astype(np.float32)).mean()  # 找到每个大于阈值的标注框。求均值得到适应度
    bpr = (best > thr).astype(np.float32).mean()  # 满足阈值的标注框所占比例
    return f, bpr

# 注意：使用别人的预训练权重时，不要冻结太多的权重。因为别人的权重是基于别人的数据集所聚类生成的anchor的，这个anchor与你的数据集不大可能适配
def main(img_size=512, n=9, thr=0.25, gen=1000):
    # 从数据集中读取所有图片的wh以及对应bboxes的wh
    dataset = VOCDataSet(voc_root="/data", year="2012", txt_name="train.txt")
    im_wh, boxes_wh = dataset.get_info()

    # 最大边缩放到img_size
    im_wh = np.array(im_wh, dtype=np.float32)
    # 注意：输入图像都缩放到512*512，再乘相对图像大小的标注框（即anchor也要同时缩放），而不是绝对大小的标注框。这样聚类的效果会更好，而不是绝对大小的标注框。
    # 如果输入是随机大小的图片，就预处理中的随机缩放修改图片和anchors就可以
    shapes = img_size * im_wh / im_wh.max(1, keepdims=True) 
    wh0 = np.concatenate([l * s for s, l in zip(shapes, boxes_wh)])  # 绝对大小的wh。 这里的s是每张图，l是每张图里所有的标注框大小（相对），s*l就变为绝对大小

    # Filter 过滤掉小目标
    i = (wh0 < 3.0).any(1).sum()
    if i:
        print(f'WARNING: Extremely small objects found. {i} of {len(wh0)} labels are < 3 pixels in size.')
    wh = wh0[(wh0 >= 2.0).any(1)]  # 只保留wh都大于等于2个像素的box

    # 原论文实现的是注释部分的欧式距离聚类，并在后面应用遗传算法看能否得到更好的结果
    # Kmeans calculation
    # print(f'Running kmeans for {n} anchors on {len(wh)} points...')
    # s = wh.std(0)  # 原文使用的是scipy中的kmeans方法，需要除以一个标准差再传入方法
    # k, dist = kmeans(wh / s, n, iter=30)  # points, mean distance
    # assert len(k) == n, print(f'ERROR: scipy.cluster.vq.kmeans requested {n} points but returned only {len(k)}')
    # k *= s # 得到聚类结果后再乘上标准差还原
    k = k_means(wh, n)

    # 按面积排序
    k = k[np.argsort(k.prod(1))]  # sort small to large
    f, bpr = anchor_fitness(k, wh, thr) # 计算一下求得的anchor适应度
    print("kmeans: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")

    # Evolve
    # 遗传算法(在kmeans的结果基础上变异mutation)
    npr = np.random
    f, sh, mp, s = anchor_fitness(k, wh, thr)[0], k.shape, 0.9, 0.1  # 适应度, anchors的shape, 变异比例, sigma
    pbar = tqdm(range(gen), desc=f'Evolving anchors with Genetic Algorithm:')  # progress bar
    for _ in pbar: # 迭代gen次
        v = np.ones(sh) # v.shape = [9,2],9是anchor个数，2是anchor的长宽
        while (v == 1).all():  # mutate until a change occurs (prevent duplicates)
            # npr.random(sh) 创建一个shape为sh的随机数，每个随机数都是【0,1】之间
            # (npr.random(sh) < mp 找到小于0.9的数值，也就是以90%的比例选取基因，被选中的为1，没被选中的为0。然后对被选中的进行变异
            # npr.randn(*sh) * s 正态分布随机数 * sigma   这里通过sigma调整正态分布结果
            # +1 调整成为均值为1的正态分布
            # .clip(0.3, 3.0) 将数值设置上下限
            v = ((npr.random(sh) < mp) * random.random() * npr.randn(*sh) * s + 1).clip(0.3, 3.0)
        kg = (k.copy() * v).clip(min=2.0)  # 得到变异后的anchors
        fg, bpr = anchor_fitness(kg, wh, thr) # 重新计算适应度
        if fg > f: # 如果大于原来的适应度，就替换原来的
            f, k = fg, kg.copy()
            pbar.desc = f'Evolving anchors with Genetic Algorithm: fitness = {f:.4f}'

    # 将变异后的按面积排序
    k = k[np.argsort(k.prod(1))]  # sort small to large
    print("genetic: " + " ".join([f"[{int(i[0])}, {int(i[1])}]" for i in k]))
    print(f"fitness: {f:.5f}, best possible recall: {bpr:.5f}")


if __name__ == "__main__":
    main()
