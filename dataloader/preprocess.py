## author: xin luo
## create: 2021.9.9
## des: simple pre-processing for the dset data (image and truth pair).

import numpy as np
import random
import cv2
from utils.geotif_io import readTiff
import threading as td
from queue import Queue

class normalize:
    '''normalization with the given per-band max and min values'''
    
    def __init__(self, max_bands, min_bands):
        '''max, min: list, values corresponding to each band'''
        self.max, self.min = max_bands, min_bands
        # 初始化时接收每个波段的最大值和最小值，并存储在实例变量中。

    def __call__(self, image):
        image_nor = []
        # 初始化一个空列表，用于存储归一化后的波段数据。

        if isinstance(self.max, int):
            # 如果最大值是一个整数（而不是列表），则为每个波段创建相同的最大值和最小值列表。
            self.max = [self.max for i in range(image.shape[-1])]
            self.min = [self.min for i in range(image.shape[-1])]

        for band in range(image.shape[-1]):
            # 遍历图像的每个波段。
            band_nor = (image[:, :, band] - self.min[band]) / (self.max[band] - self.min[band] + 0.0001)
            # 对每个波段进行归一化处理，使用给定的最大值和最小值。
            # 加上一个小的常数（0.0001）以避免除以零。

            image_nor.append(band_nor)
            # 将归一化后的波段数据添加到列表中。

        image_nor = np.array(image_nor)
        # 将列表转换为 numpy 数组。

        image_nor = np.clip(image_nor, 0., 1.)
        # 将归一化后的数据裁剪到 [0, 1] 范围内，确保所有值都在此范围内。

        return image_nor
        # 返回归一化后的图像数据。

def read_normalize(paths_img, paths_truth, max_bands, min_bands):
    ''' des: data (s1 ascending, s1 descending and truth) reading 
             and preprocessing
        input: 
            ascend image, descend image and truth image paths
            max, min: the max and min values of each band.
        return:
            scenes list and truths list
    '''
    scene_list, truth_list = [],[]
    # 初始化场景列表和真值列表，用于存储处理后的图像和标签数据。

    for i in range(len(paths_img)):
        ## --- data reading
        scene, _ = readTiff(paths_img[i])
        # 读取路径为 paths_img[i] 的图像数据，返回图像数组。

        truth, _ = readTiff(paths_truth[i])
        # 读取路径为 paths_truth[i] 的标签数据，返回标签数组。

        ## --- data normalization 
        scene = normalize(max_bands=max_bands, min_bands=min_bands)(scene)
        # 对图像数据进行归一化处理，将其缩放到指定的最大和最小波段值范围内。

        scene[np.isnan(scene)] = 0
        # 将图像数据中的 NaN 值替换为 0，以去除无效值。

        scene_list.append(scene), truth_list.append(truth)
        # 将处理后的图像和标签数据分别添加到场景列表和真值列表中。

    return scene_list, truth_list
    # 返回包含所有处理后图像和标签数据的列表。

# def crop(image, truth, size=[256]):
#     ''' numpy-based
#         des: randomly crop corresponding to specific size
#         input image and truth are np.array
#         input patch_size: (size of the cropped patch, the height and width are the same)
#     '''
#     start_h = random.randint(0, truth.shape[0]-size[0])
#     start_w = random.randint(0, truth.shape[1]-size[1])
#     patch = image[:, start_h:start_h+size[0], start_w:start_w+size[1]]
#     ptruth = truth[start_h:start_h+size[0], start_w:start_w+size[1]]
#     return patch, ptruth

class crop:
    ''' 
    numpy-based
    des: randomly crop corresponding to specific size
    input image and truth are np.array
    input size: (size of the height and width, the height and width are the same)
    '''
    
    def __init__(self, patch_size=[256]):
        self.patch_size = patch_size[0]
        # 初始化裁剪类，设置裁剪块的大小。

    def __call__(self, image, truth):
        start_h = random.randint(0, truth.shape[0] - self.patch_size)
        # 随机选择裁剪块的起始高度，确保不超出图像边界。

        start_w = random.randint(0, truth.shape[1] - self.patch_size)
        # 随机选择裁剪块的起始宽度，确保不超出图像边界。

        patch = image[:, start_h:start_h+self.patch_size, start_w:start_w + self.patch_size]
        # 从输入图像中裁剪出指定大小的图像块。

        ptruth = truth[start_h:start_h+self.patch_size, start_w:start_w + self.patch_size]
        # 从输入标签中裁剪出对应的标签块。

        return patch, ptruth
        # 返回裁剪后的图像块和标签块。

class crop_scales:
    ''' numpy-based
        des: randomly crop multiple-scale patches (from high to low)
        input patch_size: tuple or list (high -> low)
        we design a multi-thread processsing for resizing
    '''
    def __init__(self, patch_size=[2048, 512, 256], threads=True):
        self.patch_size = patch_size
        self.threads = threads

    def job_resize(self, q, band):
        band_down = cv2.resize(src=band, dsize=(256, 256), interpolation=cv2.INTER_AREA)
        q.put((band_down))

    def threads_resize(self, patch):
        patch_down = []
        q = Queue()
        threads = [td.Thread(target=self.job_resize, args=(q, patch[i])) for i in range(patch.shape[0])]
        start = [t.start() for t in threads]
        join = [t.join() for t in threads]
        for i in range(len(threads)):
            band_down = q.get()
            patch_down.append(band_down)
        patch_down = np.array(patch_down)
        return patch_down

    def __call__(self, image, truth):
        '''input image and turth are np.array'''        
        patches_group = []
        patch_high, ptruth_high = crop(patch_size=self.patch_size)(image, truth)  ## high scale
        patches_group.append(patch_high)
        for size in self.patch_size[1:]:
            start_offset = (self.patch_size[0]-size)//2
            patch_lower = patch_high[:, start_offset:start_offset+size, \
                                                start_offset:start_offset+size]
            patches_group.append(patch_lower)
        ptruth = ptruth_high[start_offset:start_offset + size, \
                                                start_offset:start_offset+size]        
        patches_group_down = []
        for patch in patches_group[:-1]:
            if self.threads:
                patch_down = self.threads_resize(patch)
            else:
                patch_down=[cv2.resize(patch[num], dsize=(self.patch_size[-1], self.patch_size[-1]), \
                                    interpolation=cv2.INTER_LINEAR) for num in range(patch.shape[0])]
            patches_group_down.append(np.array(patch_down))
        patches_group_down.append(patch_lower)

        return patches_group_down, ptruth

