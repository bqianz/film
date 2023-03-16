from torchvision.utils import save_image
import pandas as pd
from PIL import Image

from osgeo import gdal, osr, ogr

import os

import matplotlib.pyplot as plt
import matplotlib.image as pimage
import numpy as np
import torch

from gtif import *
from dataPreprocess import *

RE = 6371000 #radius of earth
chips_layer_names = []

df = pd.read_csv(r"df_unique.csv", header=[0])

project_df(df)

df.borehole = df.borehole.str.replace('//', '--')

data_root = r"C:\Users\mouju\Desktop\film\hds_old"

file = 'ld.tif'
ds = gdal.Open(os.path.join(data_root, file))
ulx, xres, xskew, uly, yskew, yres  = ds.GetGeoTransform()

band = ds.GetRasterBand(1)
arr = band.ReadAsArray()

df = crop_df_hds(df)

print(f'Number of boreholes: {len(df)})
      
chip_size = 128
      
data_root = r"geo90epsg3413_10m"
chips_root = r"raster_chips_geo90"
if not os.path.exists(chips_root):
    os.makedirs(chips_root)

file_list = os.listdir(data_root)
n_channels = len(file_list)

f = os.path.join(data_root, file_list[0])
arr, extent, _ = read_geotiff(f, 1)
ulx, xres, xskew, uly, yskew, yres = extent
preloaded = np.zeros([n_channels, arr.shape[0], arr.shape[1]])
print(f"xres = {xres}, yres = {yres}")

for i, file in enumerate(file_list):
    # name = file.split('_')[0]
    # print(name)
    f = os.path.join(data_root, file)
    arr, extent, _ = read_geotiff(f, 1)
    
    preloaded[i, :, :] = arr

for _, row in df.iterrows():
    
    x = row.at['proj_x']
    y = row.at['proj_y']
    bh = row.borehole

    x_start = np.round((x - ulx) / xres - chip_size/2).astype(int)
    x_end = x_start + chip_size

    y_start = np.round((y - uly) / yres - chip_size/2).astype(int)
    y_end = y_start + chip_sizw


    image= preloaded[:, y_start:y_end, x_start:x_end]
    # print(image.shape)
    
    with open(os.path.join(chips_source, f'{bh}.npy'), 'rb') as f:
        loaded = np.load(f)
    
    appended = np.concatenate((loaded, image), axis=0)
    
    with open(os.path.join(chips_dest, f'{bh}.npy'), 'wb') as f:
        np.save(f, appended)

      
## normalize chips
chips_root = r"C:\Users\mouju\Desktop\film\raster_chips_arcticdem"
chips_layer_names=['dem', 'insolation', 'twi']
normalize_chips(chips_root, chips_layer_names)