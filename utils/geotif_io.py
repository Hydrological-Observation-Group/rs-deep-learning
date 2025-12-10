## author: xin luo 
# creat: 2021.6.18
# modify: 2022.1.25

import numpy as np
from osgeo import gdal
from osgeo import osr

### tiff image reading
def readTiff(path_in):
    '''
    arg:
        path_in: image path
    return: 
        img: numpy array, exent: tuple, (x_min, x_max, y_min, y_max) 
        proj info, and dimentions: (row, col, band)
    '''
    RS_Data = gdal.Open(path_in)
    # 打开指定路径的 TIFF 图像文件，返回一个 GDAL 数据集对象。

    im_col = RS_Data.RasterXSize
    # 获取图像的列数（宽度）。

    im_row = RS_Data.RasterYSize
    # 获取图像的行数（高度）。

    im_bands = RS_Data.RasterCount
    # 获取图像的波段数。

    im_geotrans = RS_Data.GetGeoTransform()
    # 获取图像的地理变换参数。

    im_proj = RS_Data.GetProjection()
    # 获取图像的投影信息。

    img_array = RS_Data.ReadAsArray(0, 0, im_col, im_row)
    # 将图像数据读取为一个 numpy 数组。

    left = im_geotrans[0]
    # 计算图像左边界的地理坐标。

    up = im_geotrans[3]
    # 计算图像上边界的地理坐标。

    right = left + im_geotrans[1] * im_col + im_geotrans[2] * im_row
    # 计算图像右边界的地理坐标。

    bottom = up + im_geotrans[5] * im_row + im_geotrans[4] * im_col
    # 计算图像下边界的地理坐标。

    extent = (left, right, bottom, up)
    # 定义图像的地理范围。

    espg_code = osr.SpatialReference(wkt=im_proj).GetAttrValue('AUTHORITY', 1)
    # 获取图像的 EPSG 代码。

    img_info = {'geoextent': extent, 'geotrans': im_geotrans, 
                'geosrs': espg_code, 'row': im_row, 'col': im_col,
                'bands': im_bands}
    # 创建一个字典，存储图像的地理信息和维度信息。

    if im_bands > 1:
        img_array = np.transpose(img_array, (1, 2, 0))
        # 如果图像有多个波段，调整数组的维度顺序为 (行, 列, 波段)。

        return img_array, img_info
        # 返回图像数组和图像信息字典。
    else:
        return img_array, img_info
        # 如果图像只有一个波段，直接返回图像数组和图像信息字典。

###  .tiff image write
def writeTiff(im_data, im_geotrans, im_geosrs, path_out):
    '''
    input:
        im_data: tow dimentions (order: row, col),or three dimentions (order: row, col, band)
        im_geosrs: espg code correspond to image spatial reference system.
    '''
    im_data = np.squeeze(im_data)
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_Int16
    else:
        datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_data = np.transpose(im_data, (2, 0, 1))
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands,(im_height, im_width) = 1,im_data.shape
    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(path_out, im_width, im_height, im_bands, datatype, options=["TILED=YES", "COMPRESS=LZW"])
    if(dataset!= None):
        dataset.SetGeoTransform(im_geotrans)       # 
        dataset.SetProjection("EPSG:" + str(im_geosrs))      # 
    if im_bands > 1:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        del dataset
    else:
        dataset.GetRasterBand(1).WriteArray(im_data)
        del dataset
