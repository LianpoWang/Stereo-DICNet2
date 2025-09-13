from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torchvision.transforms as transforms
import numpy as np
import cv2 as cv
import torch

class SpeckleDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        L0 = os.path.join(self.root_dir,self.Speckles_frame.iloc[idx, 0])
        L1 = os.path.join(self.root_dir,self.Speckles_frame.iloc[idx, 1])
        R0 = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        U = os.path.join(self.root_dir,  self.Speckles_frame.iloc[idx, 4])
        V = os.path.join(self.root_dir,  self.Speckles_frame.iloc[idx, 5])
        Dispx_name = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 6])


        L0 = cv.imread(L0,cv.IMREAD_GRAYSCALE)  
        L1= cv.imread(L1,cv.IMREAD_GRAYSCALE)
        R0 = cv.imread(R0, cv.IMREAD_GRAYSCALE)

        U = np.genfromtxt(U, delimiter=',')
        V = np.genfromtxt(V, delimiter=',')
        Dispx = np.genfromtxt(Dispx_name, delimiter=',')

        L0  = L0
        L1 = L1
        R0 = R0
        u = U
        v = V
        Dispx = Dispx
        L0= L0[np.newaxis, ...]
        L1 = L1[np.newaxis, ...]
        R0 = R0[np.newaxis, ...]
        
        sample = {'L0':L0, 'L1': L1, 'R0': R0,'U':u, 'V': v,'Dispx': Dispx}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SpeckleDataset_fusion(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.Speckles_frame = pd.read_csv(csv_file,header=None)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.Speckles_frame)

    def __getitem__(self, idx):
        L0 = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 0])
        R0 = os.path.join(self.root_dir,self.Speckles_frame.iloc[idx, 1])
        L1 = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 2])
        R1 = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 3])
        U = os.path.join(self.root_dir,self.Speckles_frame.iloc[idx, 4])
        V = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 5])
        W = os.path.join(self.root_dir, self.Speckles_frame.iloc[idx, 5])

        L0 = cv.imread( L0,cv.IMREAD_GRAYSCALE)   #Ref_img = cv.imread(Ref_name,cv.IMREAD_GRAYSCALE)
        R0= cv.imread(R0,cv.IMREAD_GRAYSCALE)
        L1 = cv.imread(L1, cv.IMREAD_GRAYSCALE)  # Ref_img = cv.imread(Ref_name,cv.IMREAD_GRAYSCALE)
        R1 = cv.imread(R1, cv.IMREAD_GRAYSCALE)
        U = np.genfromtxt(U, delimiter=',')
        V = np.genfromtxt(V, delimiter=',')
        W = np.genfromtxt(W, delimiter=',')


        L0= L0[np.newaxis, ...]
        R0 = R0[np.newaxis, ...]
        L1 = L1[np.newaxis, ...]
        R1 = R1[np.newaxis, ...]
        U = U[np.newaxis, ...]
        V = V[np.newaxis, ...]
        W = W[np.newaxis, ...]

        sample = {'L0':L0, 'R0': R0,'L1':L1, 'R1': R1,'U':U, 'V':V, 'W': W}

        if self.transform:
            sample = self.transform(sample)

        return  sample

class Normalization(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        L0, L1, R0, U, V, Dispx= sample['L0'], sample['L1'], sample['R0'], sample['U'], sample['V'],sample['Dispx']
        
        self.mean = 0.0
        self.std = 255.0
        self.mean1 = 0
        self.std1 = 1

        return {'L0': torch.from_numpy((L0 - self.mean) / self.std).float(),
                'L1': torch.from_numpy((L1 - self.mean) / self.std).float(),
                'R0': torch.from_numpy((R0 - self.mean) / self.std).float(),
                'U': torch.from_numpy((U - self.mean1) / self.std1).float(),
                'V': torch.from_numpy((V - self.mean1) / self.std1).float(),
                'Dispx': torch.from_numpy((Dispx - self.mean1) / self.std1).float(),}



class Normalization_fusion(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        L0,R0,L1,R1,U,V,W= sample['L0'], sample['R0'], sample['L1'],sample['R1'],sample['U'], sample['V'],sample['W']

        self.mean = 0.0
        self.std = 255.0
        self.mean1 = 0
        self.std1 = 1

        return {'L0': torch.from_numpy((L0 - self.mean) / self.std).float(),
                'R0': torch.from_numpy((R0 - self.mean) / self.std).float(),
                'L1': torch.from_numpy((L1 - self.mean) / self.std).float(),
                'R1': torch.from_numpy((R1 - self.mean) / self.std).float(),
                'U': torch.from_numpy((U - self.mean1) / self.std1).float(),
                'V': torch.from_numpy((V - self.mean1) / self.std1).float(),
                'W': torch.from_numpy((W - self.mean1) / self.std1).float(),}




def main():
    batch_size = 2
    num_workers = 2
    transform = transforms.Compose([Normalization()])

    train_data = SpeckleDataset(csv_file='/home/xxx/stereo-data/plane/Train_data/Train_Annotation.csv', root_dir='/home/xxx/stereo-data/plane/Train_data', transform = transform)
    test_data = SpeckleDataset(csv_file='/home/xxx/stereo-data/plane/Test_data/Test_Annotation.csv', root_dir='/home/xxx/stereo-data/plane/Test_data', transform = transform)

    print('{} samples found, {} train samples and {} test samples '.format(len(test_data) + len(train_data),
                                                                           len(train_data),
                                                                           len(test_data)))
    train_loader = torch.utils.data.DataLoader(
        train_data , batch_size= batch_size,
        num_workers=num_workers, pin_memory=True, shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, shuffle=True)


if __name__ == '__main__':

    main()

