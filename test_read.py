import numpy as np
from collections import Counter
import pickle
from osgeo import gdal
import struct
import numpy as np

if __name__ == "__main__":
    tan_dataset = gdal.Open('./dataset/kxynl-WNS-WIN-20220530_000_019_cube.tiff', gdal.GA_ReadOnly)
    dataset = gdal.Open('./dataset/WHU-Hi/WHU-Hi-LongKou/WHU-Hi-LongKou.tif', gdal.GA_ReadOnly)
    dataset_array = gdal.Dataset.ReadAsArray(dataset)
    dataloader = np.reshape(dataset_array, (dataset_array.shape[0], -1))

    # read
    dataset = gdal.Open('./dataset/WHU-Hi/WHU-Hi-LongKou/WHU-Hi-LongKou.tif', gdal.GA_ReadOnly)
    label = gdal.Open('./dataset/WHU-Hi/WHU-Hi-LongKou/WHU-Hi-LongKou_gt.tif', gdal.GA_ReadOnly)
    dataset_array = gdal.Dataset.ReadAsArray(dataset)
    label_array = np.expand_dims(gdal.Dataset.ReadAsArray(label), 0)
    data = np.reshape(dataset_array, (dataset_array.shape[0], -1))
    label = np.reshape(label_array, (label_array.shape[0], -1))

    ## scale data
    all_data = []
    for i in range(data.shape[1]):
        tmp_data = data[:, i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data.append((tmp_data - tmp_mean) / tmp_std)
    ## encode label
    all_label = []
    for i in range(label.shape[1]):
        all_label.append(int(label[:, i]))
    all_label = np.array(all_label)

    # # split train val test
    # X_train, X_test, Y_train, Y_test = train_test_split(all_data, all_label, test_size=0.2, random_state=0)
    # X_val, X_test, Y_val, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=0)
    #
    # # slide and cut
    # print('before: ')
    # print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    # X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    # X_val, Y_val, pid_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride, output_pid=True)
    # X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    # print('after: ')
    # print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    #
    # # shuffle train
    # shuffle_pid = np.random.permutation(Y_train.shape[0])
    # X_train = X_train[shuffle_pid]
    # Y_train = Y_train[shuffle_pid]
    #
    # X_train = np.expand_dims(X_train, 1)
    # X_val = np.expand_dims(X_val, 1)
    # X_test = np.expand_dims(X_test, 1)


    print("Driver: {}/{}".format(dataset.GetDriver().ShortName,
                                 dataset.GetDriver().LongName))
    print("Size is {} x {} x {}".format(dataset.RasterXSize,
                                        dataset.RasterYSize,
                                        dataset.RasterCount))
    print("Projection is {}".format(dataset.GetProjection()))
    geotransform = dataset.GetGeoTransform()
    if geotransform:
        print("Origin = ({}, {})".format(geotransform[0], geotransform[3]))
        print("Pixel Size = ({}, {})".format(geotransform[1], geotransform[5]))

    band = dataset.GetRasterBand(2)
    projection = dataset.GetProjection()
    spatial = dataset.GetSpatialRef()
    print("Band Type={}".format(gdal.GetDataTypeName(band.DataType)))

    min = band.GetMinimum()
    max = band.GetMaximum()
    if not min or not max:
        (min, max) = band.ComputeRasterMinMax(True)
    print("Min={:.3f}, Max={:.3f}".format(min, max))

    if band.GetOverviewCount() > 0:
        print("Band has {} overviews".format(band.GetOverviewCount()))

    if band.GetRasterColorTable():
        print("Band has a color table with {} entries".format(band.GetRasterColorTable().GetCount()))

    scanline = band.ReadRaster(xoff=0, yoff=0,
                               xsize=band.XSize, ysize=1,
                               buf_xsize=band.XSize, buf_ysize=1,
                               buf_type=gdal.GDT_Float32)
    tuple_of_floats = struct.unpack('f' * band.XSize, scanline)
    print(type(tuple_of_floats))
    array_of_floats = np.asarray(tuple_of_floats)
    print(np.max(array_of_floats))
    print(tuple_of_floats)
    # with open('./dataset/challenge2017.pkl', 'rb') as fin:
    #     res = pickle.load(fin)
    #
    # all_data = res['data']
    # all_label = res['label']
    # print(Counter(all_label))
    # data1 = all_data[0]
    # print(data1)