import torch
import numpy as np
import random
from torch.utils.data import DataLoader, Dataset
from utils import cuda
import pickle

class TrajDataset(Dataset):
    def __init__(self, trajs, remove_or_cut, max_length = 10000000):
        super(TrajDataset, self).__init__()
        assert(remove_or_cut == 'remove' or remove_or_cut == 'cut')
        self.rawdata = trajs
        select = []
        for traj in trajs:
            if len(traj) <= max_length:
                select.append(traj)
            elif remove_or_cut == 'cut':
                select.append(traj[:max_length])
        trajs = select
        self.length = torch.tensor(list(map(len, trajs)))
        self.min = []
        self.max = []
        for i in range(len(trajs[0][0])):
            onedim = [[y[i] for y in x] for x in trajs]
            self.min.append(min(map(min, onedim)))
            self.max.append(max(map(max, onedim)))
        self.min = torch.tensor(self.min)
        self.max = torch.tensor(self.max)
        self.trajs = self.min.unsqueeze(0).unsqueeze(0).repeat(len(trajs), max_length, 1)
        for i in range(len(trajs)):
            self.trajs[i][:self.length[i]] = torch.tensor(trajs[i])
        self.trajs -= self.min
        self.trajs /= self.max - self.min
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, x):
        return self.length[x], self.trajs[x]
    def collate_fn(self, data):
        for num, one in enumerate(data):
            data[num] = list(one)
            data[num].append(num)
        data.sort(key=lambda x:-x[0])
        x, y, index = zip(*data)
        return cuda(torch.stack(x)), cuda(torch.stack(y)), index

def readfile(filename, batch_size, max_length, remove_or_cut = 'remove', shuffle = True, split = 0.0):
    arr = pickle.load(open(filename, 'rb'))
    if split != 0.0:
        arr1 = []
        arr2 = []
        index = list(range(len(arr)))
        if shuffle:
            random.shuffle(index)
        for i in range(len(arr)):
            if index[i] < split * len(arr):
                arr2.append(arr[i])
            else:
                arr1.append(arr[i])
        dataset1 = TrajDataset(arr1, remove_or_cut, max_length)
        dataset2 = TrajDataset(arr2, remove_or_cut, max_length)
        return DataLoader(dataset1, batch_size, shuffle, collate_fn=dataset1.collate_fn), DataLoader(dataset2, batch_size, shuffle, collate_fn=dataset2.collate_fn)
    dataset = TrajDataset(arr, remove_or_cut, max_length)
    return DataLoader(dataset, batch_size, shuffle, collate_fn=dataset.collate_fn)