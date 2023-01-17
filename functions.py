import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
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

from datetime import datetime

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

def filter_df_visible_ice(df):
    df['visible_ice'].replace(['None'], 'No visible ice', regex=True, inplace=True)

    ordered_ice = ['No visible ice', 'Low', "Medium to high", 'High', 'Pure ice']
    df['visible_ice'] = pd.Series(pd.Categorical(df['visible_ice'], categories=ordered_ice, ordered=True))

    return df.dropna(subset=['visible_ice'])

class Geo90Dataset(Dataset):
    def __init__(self, data_root, df, base_lat, base_lng, chip_size=32):
        
        self.base_lat = base_lat
        self.base_lng = base_lng
        
        self.df = df
        
        self.chip_size = chip_size
        
        self.trans = transforms.ToTensor()
        
        self.n_channels = len(os.listdir(data_root))
        self.preloaded = torch.ones(self.n_channels, 6000, 6000)
        
        for i, file in enumerate(os.listdir(data_root)):
            # name = file.split('_')[0]
            # print(name)
            
            I = np.array(Image.open(data_root + os.path.sep + file))
            print(I.shape)
            # I = plt.imread(data_root + os.path.sep + file)
            
#             print(I.max())
#             print(I.min())
            
#             # normalize
#             I = (((I - I.min()) / (I.max() - I.min())) * 255.9).astype(np.uint8)
#             print(I.max())
#             print(I.min())
            
            self.preloaded[i] = self.trans(I)
            
        
        
        print('Dataset initialized')
        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        bh_id = row.at['borehole']
        lat = row.at['latitude']
        lng = row.at['longitude']
        

        pixel_len = 5/6000
        

        lat_index_start = np.round((self.base_lat - lat) / pixel_len - self.chip_size/2).astype(int)
        lat_index_end = lat_index_start + self.chip_size
        
        lng_index_start = np.round((lng - self.base_lng) / pixel_len - self.chip_size/2).astype(int)
        lng_index_end = lng_index_start + self.chip_size
        
        image = self.preloaded[:, lat_index_start:lat_index_end,lng_index_start:lng_index_end]
        
        
        # surface = torch.tensor(row.filter(['depth'])).float()
        surface = torch.tensor(row.filter(['depth_norm', 'month_cyclic_norm', 'lat_norm', 'lng_norm', 'year_norm'])).float()
        
        frozen = torch.tensor(row.at['frozen']).float()
        
        # visible_ice = torch.tensor(row.at['visible_ice']).float()
        visible_ice = torch.tensor(row.at['visible_ice_code']).long()
        
        # material_ice = torch.tensor(row.at['material_ice']).float()
        
        return {'image': image, 'surface_data': surface, 'frozen': frozen,  'visible_ice': visible_ice} #'material_ice': material_ice}
    
# def train_resnet(trainloader, testloader, print_epochs = False, loss_fn = torch.nn.BCELoss(), n_channels = 1):
    
#     num_classes = 5
#     model= models.resnet18()
#     model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     model.fc = nn.Linear(in_features=512, out_features=num_classes)
    
#     model.to(device)
    
#     optimizer = torch.optim.Adam(model.parameters(), weight_decay = L2_param)

#     epoch_loss = np.zeros([train_max_epoch, 2])
#     for epoch in range(train_max_epoch):  # loop over the dataset multiple times

#         model.train()
#         running_loss_sum = 0.0
#         for i, data in enumerate(trainloader, 0): # loop over each sample
#             # get the inputs; data is a list of [inputs, labels]
#             image, labels = data['image'].to(device), data['label'].to(device)

#             predicted = model(image)
            
            
# #             print(predicted.squeeze().get_device())
# #             print('\n')
# #             print(labels.get_device())
            
            
#             # squeeze: return tensor with all dimensions of size 1 removed
#             loss = loss_fn(predicted.squeeze(), labels)
            
#             optimizer.zero_grad()

#             loss.backward()

#             optimizer.step()

#             running_loss_sum += loss.item()

#         # ----------- get validation loss for current epoch --------------
#         model.eval()
#         validation_loss_sum = 0.0
#         for i, data in enumerate(testloader, 0): # loop over each sample

#             image, labels = data['image'].to(device), data['label'].to(device)

#             predicted = model(image)
            
#             loss = loss_fn(predicted.squeeze(), labels)

#             validation_loss_sum += loss.item()

#         # ---------------- print statistics ------------------------

#         running_loss = running_loss_sum / len(trainloader)
#         validation_loss = validation_loss_sum / len(testloader)
#         epoch_loss[epoch, :] =  [running_loss, validation_loss]
        
#         if print_epochs:
#             print('epoch %2d: running loss: %.5f, validation loss: %.5f' %
#                           (epoch + 1, running_loss, validation_loss))
        
#         torch.save(model.state_dict(), os.path.join(models_dir, 'epoch-{}.pt'.format(epoch+1)))
    
#     if print_epochs:
#         print('Finished Training')
        
#     return epoch_loss
        
# def test_resnet(epoch_loss, print_model_epoch = False, n_channels = 1):
    
#     # ------ select model ---------
#     ind = np.argmin(epoch_loss[:, 1])
    
#     num_classes=5
    
#     model= models.resnet18()
#     model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#     model.fc = nn.Linear(in_features=512, out_features=num_classes)
    
#     model.load_state_dict(torch.load('{}epoch-{}.pt'.format(models_dir, ind+1)))
    
#     model.to(device)
    
#     if print_model_epoch:
#         print("epoch {} model selected".format(ind+1))
    
#     # evaluate model on test set
#     model.eval()

#     with torch.no_grad():
#         test_results = []
        
#         for i, data in enumerate(testloader, 0):
#             image, labels = data['image'].to(device), data['label'].to(device)
#             # y_test.append(label.numpy().list())
#             # print(label.shape)
#             # print(images.shape)

#             output = model(image)
            
#             test_results.extend(output)
            
#     return test_results