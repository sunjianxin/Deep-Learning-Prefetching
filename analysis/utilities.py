import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class MyTrainDataSet(Dataset):
    def __init__(self, train_dataset, train_label):
        self.train_dataset = train_dataset
        self.train_label = train_label
        
    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, idx):
        seq = self.train_dataset[idx]
        label = self.train_label[idx]
        
        return [seq, label]


class MyTestDataSet(Dataset):
    def __init__(self, test_dataset, test_label):
        self.test_dataset = test_dataset
        self.test_label = test_label

    def __len__(self):
        return len(self.test_dataset)

    def __getitem__(self, idx):
        seq = self.test_dataset[idx]
        label = self.test_label[idx]

        return [seq, label]



def show_statistic(l):
    max_value = max(l)
    min_value = min(l)
    mean = sum(l)/len(l)
    var = sum([((x - mean) ** 2) for x in l]) / len(l)
    std = var**0.5
    print("-----------------------------")
    print("min:", min_value)
    print("max:", max_value)
    print("mean:", mean)
    print("std:", std)

def normalization(l):
    max_value = max(l)
    min_value = min(l)
    mean = sum(l)/len(l)
    var = sum([((x - mean) ** 2) for x in l]) / len(l)
    std = var**0.5
    # print("min:", min_value)
    # print("max:", max_value)
    # print("mean:", mean)
    # print("std:", std)
    
    for i in range(len(l)):
        l[i] = (l[i] - mean)/std
    
    return l
    
def normalize_one(d, mean, std):
    # do normalization
    for i in range(len(d)):
        d[i] = (d[i] - mean)/std
    
def min_max_scaling(l):
    max_value = max(l)
    min_value = min(l)
    for i in range(len(l)):
        l[i] = (l[i] - min_value)/(max_value - min_value)
    return min_value, max_value

def min_max_scaling_radius(l, radius):
    max_value = max(l)
    min_value = min(l)
    for i in range(len(l)):
        l[i] = (l[i] - min_value)/(max_value - min_value)*radius*2 - radius
    return min_value, max_value

def min_max_unscaling(l, min_value, max_value):
    for i in range(len(l)):
        l[i] = l[i] * (max_value - min_value) + min_value
    
def load_data(file_name, test_size):
    f = open(file_name)
    df = pd.read_csv(f)
    data = np.array(df[['x_0','x']])
    x = data[:,0].tolist()
    x.append(data[-1, -1])
    # do normalization
    # show_statistic(x)
    min_max_scaling(x)
    # show_statistic(x)
    
    data = np.array(df[['y_0','y']])
    y = data[:,0].tolist()
    y.append(data[-1, -1])
    # do normalization
    min_max_scaling(y)

    data = np.array(df[['z_0','z']])
    z = data[:,0].tolist()
    z.append(data[-1, -1])
    # do normalization
    min_max_scaling(z)
    
    train_set_x = x[:-test_size]
    test_set_x = x[-test_size:]
    train_set_y = y[:-test_size]
    test_set_y = y[-test_size:]
    train_set_z = z[:-test_size]
    test_set_z = z[-test_size:]
    
    return train_set_x, test_set_x, train_set_y, test_set_y, train_set_z, test_set_z

def load_data_2(file_name, test_size):
    f = open(file_name)
    df = pd.read_csv(f)
    data = np.array(df[['x_0','x']])
    x = data[:,0].tolist()
    x.append(data[-1, -1])
    # do normalization
    # show_statistic(x)
    # min_max_scaling(x)
    # show_statistic(x)
    
    data = np.array(df[['y_0','y']])
    y = data[:,0].tolist()
    y.append(data[-1, -1])
    # do normalization
    # min_max_scaling(y)

    data = np.array(df[['z_0','z']])
    z = data[:,0].tolist()
    z.append(data[-1, -1])
    # do normalization
    # min_max_scaling(z)
    
    train_set_x = x[:-test_size]
    test_set_x = x[-test_size:]
    train_set_y = y[:-test_size]
    test_set_y = y[-test_size:]
    train_set_z = z[:-test_size]
    test_set_z = z[-test_size:]
    
    return train_set_x, test_set_x, train_set_y, test_set_y, train_set_z, test_set_z

def load_data_3(file_name, test_size):
    f = open(file_name)
    df = pd.read_csv(f)
    
    data = np.array(df[['x']])
    x = data[:,0].tolist()
    # do normalization
    # show_statistic(x)
    # min_max_scaling(x)
    # show_statistic(x)
    
    data = np.array(df[['y']])
    y = data[:,0].tolist()
    # do normalization
    # min_max_scaling(y)

    data = np.array(df[['z']])
    z = data[:,0].tolist()
    # do normalization
    # min_max_scaling(z)
    
    train_set_x = x[:-test_size]
    test_set_x = x[-test_size:]
    train_set_y = y[:-test_size]
    test_set_y = y[-test_size:]
    train_set_z = z[:-test_size]
    test_set_z = z[-test_size:]
    
    return train_set_x, test_set_x, train_set_y, test_set_y, train_set_z, test_set_z

def load_test_data(file_name):
    f = open(file_name)
    df = pd.read_csv(f)
    
    data = np.array(df['x'])
    x = data.tolist()
    
    # print("-----------------------------")
    # show_statistic(x)
    # x = normalization(x)
    # min_max_scaling(x)
    # show_statistic(x)

    data = np.array(df['y'])
    y = data.tolist()
    # y = normalization(y)
    # min_max_scaling(x)
    # print("y statistics:")
    # show_statistic(y)

    data = np.array(df['z'])
    z = data.tolist()
    # z = normalization(z)
    # min_max_scaling(x)
    # print("z statistics:")
    # show_statistic(z)
    
    return x, y, z

def normalize_all(d_1, d_2, d_3, d_4, d_5, l_1, l_2, l_3, l_4, l_5):
    d_all = d_1 + d_2 + d_3 + d_4 + d_5
    # get mean and std of all traning data
    mean = sum(d_all)/len(d_all)
    var = sum([((x - mean) ** 2) for x in d_all]) / len(d_all)
    std = var**0.5
    # do normalization for each training data set
    for i in range(len(d_1)):
        d_1[i] = (d_1[i] - mean)/std
    for i in range(len(d_2)):
        d_2[i] = (d_2[i] - mean)/std
    for i in range(len(d_3)):
        d_3[i] = (d_3[i] - mean)/std
    for i in range(len(d_4)):
        d_4[i] = (d_4[i] - mean)/std
    for i in range(len(d_5)):
        d_5[i] = (d_5[i] - mean)/std
        
    # do normalization for each test data set by using mean and std from traning data sets
    for i in range(len(l_1)):
        l_1[i] = (l_1[i] - mean)/std
    for i in range(len(l_2)):
        l_2[i] = (l_2[i] - mean)/std
    for i in range(len(l_3)):
        l_3[i] = (l_3[i] - mean)/std
    for i in range(len(l_4)):
        l_4[i] = (l_4[i] - mean)/std
    for i in range(len(l_5)):
        l_5[i] = (l_5[i] - mean)/std
    # return d_1, d_2, d_3, d_4, d_5, l_1, l_2, l_3, l_4, l_5
    return mean, std


# construct list of input and label pairs
def input_data(seq, ws):
    out = []
    L = len(seq)
    
    for i in range(L - ws):
        window = seq[i:i+ws]
        label = seq[i+ws:i+ws+1]
        out.append((window, label))
        
    return out

def get_tensor(sample_size, data_x, data_y, data_z):
    for i in range(sample_size):
        # construct dataset
        x = np.asarray(data_x[i][0]).reshape(1,-1) # 1 x window_size
        y = np.asarray(data_y[i][0]).reshape(1,-1) # 1 x window_size
        z = np.asarray(data_z[i][0]).reshape(1,-1) # 1 x window_size
        xyz = np.append(np.append(x, y, axis = 0), z, axis = 0) # 3 x window_size
        xyz = np.transpose(xyz) # window_size x 3
        window_size = xyz.shape[0]
        # print("window_size", window_size)
        xyz = xyz.reshape(1, window_size, 3) # 1 x window_size x 3
        if (i == 0):
            prev_xyz = xyz
        else:
            prev_xyz = np.append(prev_xyz, xyz, axis = 0)
        
        #construct label
        x_label = np.asarray(data_x[i][1]).reshape(1,-1) # 1 x 1
        y_label = np.asarray(data_y[i][1]).reshape(1,-1) # 1 x 1
        z_label = np.asarray(data_z[i][1]).reshape(1,-1) # 1 x 1
        xyz_label = np.append(np.append(x_label, y_label, axis = 0), z_label, axis = 0) # 3 x 1
        xyz_label = xyz_label.reshape(1, 3) # 1 x 3 x 1
        if (i == 0):
            prev_xyz_label = xyz_label
        else:
            prev_xyz_label = np.append(prev_xyz_label, xyz_label, axis = 0)
            
    return prev_xyz, prev_xyz_label

# make test_dataset
def construct_test_tensor(test_set_x, test_set_y, test_set_z, window_size):
    # sequence to data sample
    test_data_x = input_data(test_set_x, window_size)
    test_data_y = input_data(test_set_y, window_size)
    test_data_z = input_data(test_set_z, window_size)
    
    
    # reconstruct test dataset and label
    sample_size = len(test_data_x)
    test_dataset, test_label = get_tensor(sample_size, test_data_x, test_data_y, test_data_z)
    
    test_dataset = test_dataset.astype("float32")
    test_label = test_label.astype("float32")
    
    return test_dataset, test_label

# make train_dataset
def construct_train_valid_tensor(train_set_x, train_set_y, train_set_z, test_set_x, test_set_y, test_set_z, window_size):
    # sequence to data sample
    train_data_x = input_data(train_set_x, window_size)
    train_data_y = input_data(train_set_y, window_size)
    train_data_z = input_data(train_set_z, window_size)
    test_data_x = input_data(test_set_x, window_size)
    test_data_y = input_data(test_set_y, window_size)
    test_data_z = input_data(test_set_z, window_size)
    
    
    # reconstruct train/test dataset and label
    sample_size = len(train_data_x)
    train_dataset, train_label = get_tensor(sample_size, train_data_x, train_data_y, train_data_z)
    sample_size = len(test_data_x)
    test_dataset, test_label = get_tensor(sample_size, test_data_x, test_data_y, test_data_z)
    
    train_dataset = train_dataset.astype("float32")
    train_label = train_label.astype("float32")
    test_dataset = test_dataset.astype("float32")
    test_label = test_label.astype("float32")
    
    return train_dataset, train_label, test_dataset, test_label
