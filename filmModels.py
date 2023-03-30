import torch
import os
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.models as models
import time

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support

class filmDataset(Dataset):
    def __init__(self, chips_root, df, tab_params, label, chip_size=64):
        
        self.df = df
        self.chip_size = chip_size
        self.tab_params = tab_params
        self.label = label
        self.chips_root = chips_root  
            
        file_list = os.listdir(chips_root)
        with open(os.path.join(chips_root, file_list[0]), 'rb') as f:
            loaded = np.load(f)
        
        self.n_channels = loaded.shape[0]
        print("Dataset contains {} channels".format(self.n_channels))

        
    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        
        row = self.df.iloc[idx]
        bh_id = row.at['borehole']
#         lat = row.at['latitude']
#         lng = row.at['longitude']
        
        x = row.at['proj_x']
        y = row.at['proj_y']
        
        with open(os.path.join(self.chips_root, f'{bh_id}.npy', ), 'rb') as f:
            loaded = np.load(f)
        
        if self.chip_size > loaded.shape[1]:
            raise IndexError(f'chip_size parameter {self.chip_size} bigger than size of prepared chips {loaded.shape[1]} in {self.chips_root}')
        
        start = int(loaded.shape[1] / 2) - int(self.chip_size/2)
        end = start + self.chip_size
        
        image= torch.tensor(loaded[:, start:end, start:end]).float()
        
        tabular_parameters = torch.tensor(row.filter(self.tab_params)).float()
        
        # visible_ice = torch.tensor(row.at['visible_ice']).float()
        label_value = torch.tensor(row.at[self.label]).long()
        
        # material_ice = torch.tensor(row.at['material_ice']).float()
        
        
        return {'image': image, 'tab_params': tabular_parameters, 'label': label_value}

def split_data(full_dataset):
    train_size = int(0.8 * len(full_dataset))
    test_size = len(full_dataset) - train_size

    # batchsize can cause error when last leftover batchsize is 1, batchnorm cannot function on 1 sample data
    batchsize = 20
    while(train_size % batchsize == 1):
        batchsize+=1
    print(f'Batch size {batchsize}')

    train_data, test_data = torch.utils.data.random_split(full_dataset, [train_size, test_size]) #generator=torch.Generator().manual_seed(42)

    trainloader = DataLoader(train_data, batch_size=batchsize, shuffle=True)
    testloader = DataLoader(test_data, batch_size=batchsize, shuffle=True)
    print(f'Data split into training size {train_size} and testing size {test_size}')
    
    return trainloader, testloader

def iterate(meta_params, trainloader, testloader):
    
    max_iterations = meta_params['max_iterations']
    n_classes = meta_params['n_classes']
    n_channels = meta_params['n_channels']
    film = meta_params['film']
    
    results = np.zeros([max_iterations, n_classes*4 + 1])

    for it in range(max_iterations):
        start = time.time()
        
        if film:
            epoch_loss = train_film(meta_params, trainloader, testloader)
            acc, scores, gen_model, net_model = test_film(meta_params, testloader, epoch_loss)

        else:
            epoch_loss_mlp = train_mlp(meta_params, trainloader, testloader)
            acc, scores = test_mlp(meta_params, testloader, epoch_loss_mlp)
    
        # scores = precision, recall, fscore, support
        results[it, 0] = acc

        for j, score in enumerate(scores):
            start_ind = 1 + j*n_classes
            results[it, start_ind: start_ind + n_classes] = score

        end = time.time()

        print('iteration {} elapsed time: {}, accuracy : {}'.format(it+1, end-start, acc))
    
    return results

def test_film(meta_params, testloader, epoch_loss):
    
    device = meta_params['device']
    hidden_width = meta_params['hidden_width']
    hidden_nblocks = meta_params['hidden_nblocks']
    n_channels = meta_params['n_channels']
    output_size = meta_params['n_classes']
    print_model_epoch = meta_params['print_test_model']
    full_dataset = meta_params['full_dataset']
    
    # ------ select model ---------
    ind = np.argmin(epoch_loss[:, 1])
    
    n_film_params = hidden_width * hidden_nblocks * 2
    
    generator = models.resnet18()
    generator.fc = torch.nn.Linear(512, n_film_params)
    generator.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    gen_model = generator

    input_size = list(full_dataset[0]['tab_params'].size())
    net_model = mlp_film(meta_params, input_size[0], output_size)
    
    model_path = r'models\film'
    
    gen_model.load_state_dict(torch.load(os.path.join(model_path, 'gen-epoch-{}.pt'.format(ind+1))))
    net_model.load_state_dict(torch.load(os.path.join(model_path, 'net-epoch-{}.pt'.format(ind+1))))
    
    gen_model.to(device)
    net_model.to(device)
    
    if print_model_epoch:
        print("epoch {} model selected".format(ind+1))
    
    # evaluate model on test set
    gen_model.eval()
    net_model.eval()
    with torch.no_grad():
        y_test = []
        y_pred = []
        for i, data in enumerate(testloader, 0):
            images, surface_data, labels = data['image'].to(device), data['tab_params'].to(device), data['label'].to(device)

            # y_test.append(label.numpy().list())
            # print(label.shape)
            # print(images.shape)

            gen_params = gen_model(images)
            predicted = net_model(surface_data, gen_params)
            
            sm = torch.nn.Softmax(dim=-1)
            output = sm(predicted)
            # print(output)
            
            
            max_results = torch.max(output, dim= -1)
            predicted = max_results.indices
            #predicted = torch.round(output)
            # print(predicted.shape)
            
            y_test.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            
#             predicted = torch.squeeze(predicted)

#             predicted = torch.round(predicted)
#             # print(predicted.shape)
            
            
            
#             lb = labels.tolist()
#             pr = predicted.tolist()
#             y_test.extend(lb)
#             y_pred.extend(pr)
    
    arr_accuracy = accuracy_score(y_test, y_pred)
    scores = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    return arr_accuracy, scores, gen_model, net_model


#     print(confusion_matrix(y_test,y_pred))
#     print(classification_report(y_test,y_pred))
#     print(accuracy_score(y_test, y_pred))

## Pure MLP

def train_film(meta_params, trainloader, testloader):
    device = meta_params['device']
    hidden_width = meta_params['hidden_width']
    hidden_nblocks = meta_params['hidden_nblocks']
    n_film_params = hidden_width * hidden_nblocks * 2
    n_channels = meta_params['n_channels']
    output_size = meta_params['n_classes']
    print_epochs = meta_params['print_train_progress']
    train_max_epoch = meta_params['train_max_epoch']
    full_dataset = meta_params['full_dataset']
    L2_param = meta_params['L2_param']
    loss_fn = meta_params['loss_fn']
    
    generator = models.resnet18()
    generator.fc = torch.nn.Linear(512, n_film_params)
    generator.conv1 = torch.nn.Conv2d(n_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    
    gen_model = generator

    # print(gen_model)

    input_size = list(full_dataset[0]['tab_params'].size())
    net_model = mlp_film(meta_params, input_size[0],output_size).to(device)
    
    gen_optimizer = torch.optim.Adam(gen_model.parameters(), weight_decay = L2_param)
    net_optimizer = torch.optim.Adam(net_model.parameters(), weight_decay = L2_param)
    
    gen_model.to(device)
    net_model.to(device)

    # --------- check back propagation ----------- -
    # net_model.fc1.weight.register_hook(lambda x: print('grad accumulated in mlp fc1'))
    # gen_first_layer = gen_model.encoder.blocks[0].blocks[0].blocks[0].conv
    # gen_first_layer.weight.register_hook(lambda x: print('grad accumulated in resnet first layer'))

    epoch_loss = np.zeros([train_max_epoch, 2])
    for epoch in range(train_max_epoch):  # loop over the dataset multiple times

        # ------------ train -----------------
        gen_model.train()
        net_model.train()
        running_loss_sum = 0.0
        for i, data in enumerate(trainloader, 0): # loop over each sample

            # get the inputs; data is a list of [inputs, labels]
            images, surface_data, labels = data['image'].to(device), data['tab_params'].to(device), data['label'].to(device)

            gen_params = gen_model(images)
            predicted = net_model(surface_data, gen_params)
            predicted = torch.squeeze(predicted)
            loss = loss_fn(predicted, labels)

            gen_optimizer.zero_grad()
            net_optimizer.zero_grad()

            loss.backward()

            gen_optimizer.step()
            net_optimizer.step()

            running_loss_sum += loss.item()

        # ----------- get validation loss for current epoch --------------
        gen_model.eval()
        net_model.eval()
        validation_loss_sum = 0.0
        for i, data in enumerate(testloader, 0): # loop over each sample

            # get the inputs; data is a list of [inputs, labels]
            images, surface_data, labels = data['image'].to(device), data['tab_params'].to(device), data['label'].to(device)

            # TODO: exammine film_params gradients / readup pytorch
            gen_params = gen_model(images)
            predicted = net_model(surface_data, gen_params)
            predicted = torch.squeeze(predicted)
            loss = loss_fn(predicted, labels)
            validation_loss_sum += loss.item()

        # ---------------- print statistics ------------------------

        running_loss = running_loss_sum / len(trainloader)
        validation_loss = validation_loss_sum / len(testloader)
        epoch_loss[epoch, :] =  [running_loss, validation_loss]
        
        if print_epochs:
            print('epoch %2d: running loss: %.5f, validation loss: %.5f' %
                          (epoch + 1, running_loss, validation_loss))
        
        model_path = r'models\film'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(gen_model.state_dict(), os.path.join(model_path, 'gen-epoch-{}.pt'.format(epoch+1)))
        torch.save(net_model.state_dict(), os.path.join(model_path, 'net-epoch-{}.pt'.format(epoch+1)))

    if print_epochs:
        print('Finished Training')
    
    return epoch_loss


class mlp_film(torch.nn.Module):
        def __init__(self, meta_params, input_size, output_size = 1):
            super(mlp_film, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            
            self.hidden_width = meta_params['hidden_width']
            self.hidden_nblocks = meta_params['hidden_nblocks']
            
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_width)
            self.fc2 = torch.nn.Linear(self.hidden_width,self.hidden_width)
            self.fc3 = torch.nn.Linear(self.hidden_width, self.output_size)
            
            self.relu = torch.nn.ReLU()
            self.end= torch.nn.Softmax(dim = -1) ## sigmoid for multi-label, softmax for multi-class (mutually exclusive)
            
            self.dropout = torch.nn.Dropout(0.25)
            
        def forward(self, x, film_params):
            out = self.fc1(x)
            out = self.relu(out)
            
            
            for i in range(self.hidden_nblocks):
                out = self.fc2(out)
                
                # ------- film layer -----------
                start = i * self.hidden_width * 2
                mid = start + self.hidden_width
                end = mid + self.hidden_width
                
                gamma = film_params[:, start : mid]
                beta = film_params[:, mid : end]
                
#                 print(out.shape)
#                 print(gamma.shape)
#                 print(beta.shape)
                
                out = out * gamma
                out += beta
                # ------- film layer -----------
                # out = self.dropout(out)
                out = self.relu(out)
            
            out = self.fc3(out)
            # out = self.end(out)
            return out

        
class mlp_pure(torch.nn.Module):
        def __init__(self, meta_params, input_size, output_size):
            super(mlp_pure, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            
            self.hidden_size = meta_params['hidden_width']
            self.hidden_nblocks = meta_params['hidden_nblocks']
            
            self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size)
            self.fc2 = torch.nn.Linear(self.hidden_size,self.hidden_size)
            self.fc3 = torch.nn.Linear(self.hidden_size, self.output_size)
            
            self.relu = torch.nn.ReLU()
            self.end = torch.nn.Softmax(dim=-1) ## sigmoid for multi-label, softmax for multi-class (mutually exclusive)
            
        def forward(self, x):
            out = self.fc1(x)
            out = self.relu(out)
            # print(out.shape)
            
            for i in range(self.hidden_nblocks):
                out = self.fc2(out)
                out = self.relu(out)
            
            out = self.fc3(out)
            # out = self.end(out)
            return out

def train_mlp(meta_params, trainloader, testloader):
    device = meta_params['device']
    hidden_width = meta_params['hidden_width']
    hidden_nblocks = meta_params['hidden_nblocks']
    n_film_params = hidden_width * hidden_nblocks * 2
    n_channels = meta_params['n_channels']
    output_size = meta_params['n_classes']
    print_epochs = meta_params['print_train_progress']
    train_max_epoch = meta_params['train_max_epoch']
    full_dataset = meta_params['full_dataset']
    L2_param = meta_params['L2_param']
    loss_fn = meta_params['loss_fn']
    
    
    input_size = list(full_dataset[0]['tab_params'].size())
    surface_model = mlp_pure(meta_params, input_size[0],output_size)
    
    surface_model.to(device)
    
    optimizer = torch.optim.Adam(surface_model.parameters(), weight_decay = L2_param)

    epoch_loss = np.zeros([train_max_epoch, 2])
    for epoch in range(train_max_epoch):  # loop over the dataset multiple times

        surface_model.train()
        running_loss_sum = 0.0
        for i, data in enumerate(trainloader, 0): # loop over each sample

            # get the inputs; data is a list of [inputs, labels]
            surface_data, labels = data['tab_params'].to(device), data['label'].to(device)

            predicted = surface_model(surface_data)
            
#             print(predicted.squeeze().shape)
#             print(labels.shape)
            
            
            # squeeze: return tensor with all dimensions of size 1 removed
            loss = loss_fn(predicted.squeeze(), labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss_sum += loss.item()

        # ----------- get validation loss for current epoch --------------
        surface_model.eval()
        validation_loss_sum = 0.0
        for i, data in enumerate(testloader, 0): # loop over each sample

            surface_data, labels = data['tab_params'].to(device), data['label'].to(device)

            predicted = surface_model(surface_data)
            
            loss = loss_fn(predicted.squeeze(), labels)

            validation_loss_sum += loss.item()

        # ---------------- print statistics ------------------------

        running_loss = running_loss_sum / len(trainloader)
        validation_loss = validation_loss_sum / len(testloader)
        epoch_loss[epoch, :] =  [running_loss, validation_loss]
        
        if print_epochs:
            print('epoch %2d: running loss: %.5f, validation loss: %.5f' %
                          (epoch + 1, running_loss, validation_loss))
        
        model_path = r'models\mlp'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(surface_model.state_dict(), os.path.join(model_path, 'epoch-{}.pt'.format(epoch+1)))
    
    if print_epochs:
        print('Finished Training')
        
    return epoch_loss
        
def test_mlp(meta_params, testloader, epoch_loss):
    device = meta_params['device']
    hidden_width = meta_params['hidden_width']
    hidden_nblocks = meta_params['hidden_nblocks']
    n_channels = meta_params['n_channels']
    output_size = meta_params['n_classes']
    print_model_epoch = meta_params['print_test_model']
    full_dataset = meta_params['full_dataset']
    
    # ------ select model ---------
    ind = np.argmin(epoch_loss[:, 1])
    
    input_size = list(full_dataset[0]['tab_params'].size())
    
    surface_model = mlp_pure(meta_params, input_size[0],output_size)
    
    model_path = r'models\mlp'
    surface_model.load_state_dict(torch.load(os.path.join(model_path, 'epoch-{}.pt'.format(ind+1))))
    
    surface_model.to(device)
    
    if print_model_epoch:
        print("epoch {} model selected".format(ind+1))
    
    # evaluate model on test set
    surface_model.eval()

    with torch.no_grad():
        y_test = []
        y_pred = []
        y_cert = []
        for i, data in enumerate(testloader, 0):
            surface_data, labels = data['tab_params'].to(device), data['label'].to(device)

            output = surface_model(surface_data)
            
            sm = torch.nn.Softmax(dim=-1)
            output = sm(output)
            # print(output)
            
            
            max_results = torch.max(output, dim= -1)
            predicted = max_results.indices
            certainty = max_results.values
            #predicted = torch.round(output)
            # print(predicted.shape)
            
            y_test.extend(labels.tolist())
            y_pred.extend(predicted.tolist())
            y_cert.extend(certainty.tolist())
            


#     with open("mlp-certainty/iteration_{}.txt".format(it), "wb") as fp:   #Pickling
#         pickle.dump(y_cert, fp)
    arr_accuracy = accuracy_score(y_test, y_pred)
    scores = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    return arr_accuracy, scores
