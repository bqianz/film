import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
import torchvision.models as models
from PIL import Image
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sb

from functools import partial
from collections import OrderedDict

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import time
import pickle

# scaler = MinMaxScaler()
def normalize(values):
    # zero mean, unit variance
    value_mean = values.mean()
    value_std = values.std()
    return (values-values_mean)/values_std

def normalize_maxmin(values):
    # range from 0 to 1
    (values-values.min())/(values.max()-values.min())


def get_scaler(data):
    scaler = StandardScaler()
    print(data)
    scaler.fit(data)
    return scaler
    
def preprocess_df(df):
    # convert timecodes to year and month columns
    datetimes = pd.to_datetime(df['time'])
    df['month'] = datetimes.dt.month
    df['year'] = datetimes.dt.year

    df['month_cyclic'] = 7 - abs(df['month'] - 7)
    
    data = df[['latitude', 'longitude', 'depth', 'year', 'month_cyclic']]
    scaler = StandardScaler()
    scaler.fit(data)
    df[['lat_norm', 'lng_norm', 'depth_norm', 'year_norm', 'month_cyclic_norm']] = scaler.transform(df[['latitude', 'longitude', 'depth', 'year', 'month_cyclic']])
    
    return scaler


#     df['lat_norm'] = normalize(df['latitude'])
#     df['lng_norm'] = normalize(df['longitude'])
#     df['depth_norm'],  = normalize(df['depth'])
#     df['year_norm'] = normalize(df['year'])
#     df['month_cyclic_norm'] = normalize(df['month_cyclic'])

df = pd.read_csv('data_stephen_fix_header.csv', header=[0])
scaler = preprocess_df(df)
    
# print(df.shape[0])
# print(df['borehole'].nunique())
# df.head()

df['visible_ice'].replace(['None'], 'No visible ice', regex=True, inplace=True)

ordered_ice = ['No visible ice', 'Low', "Medium to high", 'High', 'Pure ice']
df['visible_ice'] = pd.Series(pd.Categorical(df['visible_ice'], categories=ordered_ice, ordered=True))

df2 = df.dropna(subset=['visible_ice'])

# df2.tail()

# check None values have been replaced
# len(df2[df2['visible_ice'] == 'None'])

df2['visible_ice_code'] =  df2['visible_ice'].cat.codes
# print(df2['visible_ice'].unique())
# print(df2['visible_ice_code'].unique())
# df2['visible_ice']

from matplotlib import cm

x = df2['longitude']
y = df2['latitude']
z = df2['depth']

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
p = ax.scatter(x, y, z, c=df2['visible_ice_code'], cmap=plt.cm.inferno, edgecolor='none', alpha=0.5, s=8)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_zlabel('Depth')

# ax.set_title('Visible Ice Borehole Data')

plt.gca().invert_zaxis()
# p.set_edgecolor("none")
plt.colorbar(p, ax=ax)
# plt.clim(0,4)
# fig.colorbar(pos, ax = ax)
plt.show()