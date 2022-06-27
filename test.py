import torch
import os
import cv2
import utils.transforms as T
from utils.engine import train_one_epoch, evaluate
import utils.utils as utils
import torchvision
import argparse
import numpy as np
from collections import Counter
import pickle
from dataset.dataloaders import TestDataset
from model.resnet1d import ResNet1D
from sklearn.metrics import classification_report
from utils.utils import read_data_hsi_4_with_val, read_data_physionet_4_with_val
from osgeo import gdal, gdal_array
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import glob


IMG_FORMATS = ['tif', 'tiff']

@torch.no_grad()
def evaluate(model, data_loader, device):
    prog_iter_test = tqdm(data_loader, desc="Testing", leave=False)
    all_pred_prob = []
    for batch_idx, batch in enumerate(prog_iter_test):
        input_x = batch.to(device)
        pred = model(input_x)
        all_pred_prob.append(pred.cpu().data.numpy())
    all_pred_prob = np.concatenate(all_pred_prob)
    all_pred = np.argmax(all_pred_prob, axis=1)
    return all_pred

def arr2raster(arr, raster_file, prj=None, trans=None):
    """
    将数组转成栅格文件写入硬盘
    :param arr: 输入的mask数组 ReadAsArray()
    :param raster_file: 输出的栅格文件路径
    :param prj: gdal读取的投影信息 GetProjection()，默认为空
    :param trans: gdal读取的几何信息 GetGeoTransform()，默认为空
    :return:
    """

    driver = gdal.GetDriverByName('GTiff')
    dst_ds = driver.Create(raster_file, arr.shape[1], arr.shape[0], 1, gdal.GDT_Byte)

    if prj:
        dst_ds.SetProjection(prj)
    if trans:
        dst_ds.SetGeoTransform(trans)

    # 将数组的各通道写入图片
    dst_ds.GetRasterBand(1).WriteArray(arr)

    dst_ds.FlushCache()
    dst_ds = None
    print("successfully convert array to raster")


def colormap(n):
    '''
    这里有19个颜色，根据自己数据来使用，也可以根据自己的喜爱修改颜色的值。n就是num_classes.
    np.uint8意味着array的值为0-255中间。
    cmap就是一个映射，当标签为1的时候，这个像素转换成RGB就是[244, 35,232]的三通道值。
    '''
    cmap = np.zeros([n, 3]).astype(np.uint8)
    cmap[0, :] = np.array([0, 0, 0])
    cmap[1, :] = np.array([244, 35, 232])
    cmap[2, :] = np.array([70, 70, 70])
    cmap[3, :] = np.array([102, 102, 156])
    cmap[4, :] = np.array([190, 153, 153])
    cmap[5, :] = np.array([153, 153, 153])

    cmap[6, :] = np.array([250, 170, 30])
    cmap[7, :] = np.array([220, 220, 0])
    cmap[8, :] = np.array([107, 142, 35])
    cmap[9, :] = np.array([152, 251, 152])
    cmap[10, :] = np.array([70, 130, 180])

    cmap[11, :] = np.array([220, 20, 60])
    cmap[12, :] = np.array([119, 11, 32])
    cmap[13, :] = np.array([0, 0, 142])
    cmap[14, :] = np.array([0, 0, 70])
    cmap[15, :] = np.array([0, 60, 100])

    cmap[16, :] = np.array([0, 80, 100])
    cmap[17, :] = np.array([0, 0, 230])
    cmap[18, :] = np.array([255, 0, 0])
    return cmap

def Colorize(pred_array, class_num=10):
    cmap = colormap(256)
    cmap[class_num] = cmap[-1]
    cmap = cmap[:class_num]
    color_image = np.zeros((pred_array.shape[0], pred_array.shape[1], 3))
    for label in range(0, len(cmap)):
        mask = pred_array == label
        color_image[mask][0] = cmap[label][0]
        color_image[mask][1] = cmap[label][1]
        color_image[mask][2] = cmap[label][2]

    return color_image




if __name__ == '__main__':
    # read hsi
    # label = gdal.Open('./dataset/WHU-Hi/WHU-Hi-LongKou/WHU-Hi-LongKou_gt.tif', gdal.GA_ReadOnly)
    # label_array = gdal.Dataset.ReadAsArray(label)
    path = './dataset/'
    p = str(Path(path).absolute())  # os-agnostic absolute path
    if '*' in p:
        files = sorted(glob.glob(p, recursive=True))  # glob
    elif os.path.isdir(p):
        files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
    elif os.path.isfile(p):
        files = [p]  # files
    else:
        raise Exception(f'ERROR: {p} does not exist')

    images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]

    for im_path in images:
        dataset = gdal.Open(im_path, gdal.GA_ReadOnly)
        dataset_array = gdal.Dataset.ReadAsArray(dataset)
        data = np.reshape(dataset_array, (dataset_array.shape[0], -1))
        name = Path(im_path).stem

        ## scale data
        all_data = []
        for i in range(data.shape[1]):
            tmp_data = data[:, i]
            tmp_std = np.std(tmp_data)
            tmp_mean = np.mean(tmp_data)
            all_data.append((tmp_data - tmp_mean) / tmp_std)

        test_data = np.array(all_data)
        test_data = np.expand_dims(test_data, 1)

        test_dataset = TestDataset(test_data)
        data_loader_test = torch.utils.data.DataLoader(test_dataset, batch_size=32)

        device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
        model = torch.load('./runs/classification/Resnet1dAF.pkl')
        model.to(device)
        model.eval()

        all_pred = evaluate(model, data_loader_test, device)
        plot_pred = np.reshape(all_pred, (dataset_array.shape[1], dataset_array.shape[2]))
        plt.imshow(plot_pred)
        plt.show()
        np.save('./dataset/result.npy', plot_pred)
        np.savetxt('./dataset/result.csv', plot_pred, fmt="%i", delimiter=',')

        ar, num = np.unique(plot_pred, return_counts=True)
        plot_mask = Colorize(plot_pred, len(ar))
        cv2.imwrite('./dataset/pred.jpg', plot_mask)

        raster_file = './dataset/pred.tif'  # 输出的栅格文件路径
        projection = dataset.GetProjection()
        transform = dataset.GetGeoTransform()
        arr2raster(plot_pred, raster_file, prj=projection, trans=transform)

