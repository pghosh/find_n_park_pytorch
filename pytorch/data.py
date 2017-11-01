import glob

import numpy as np
import pandas as pd
from skimage import transform
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data.dataset import Dataset

import pytorch.utils as utils

IMG_HEIGHT = 1600
IMG_WIDTH = 1200
class MasterData():
    '''
    Class to load and clean data from csv
    '''

    def __init__(self, csv_name="",base_dir="data", img_dir="images"):
        all_df = pd.read_csv("{}/{}.csv".format(base_dir,csv_name))

        '''
        load all identifiers from the data directory. The advantage of doing that here is we don't have to 
        be worried about mismatched indexes etc. The pre process step downloads all good data files and stores 
        in the data location
        '''
        #Todo: clean and load image file names here

        self.X = []
        y = []

        self.le = MultiLabelBinarizer()
        self.y = self.le.fit_transform(y)

        # This code is for single label
        #self.y = self.y[:, utils.get_idx(class_name,self.le)]


    def train_val_split(self, val_size=0.1):
        '''
        Since dataset class knows best about the data this is a good place to put train test split
        That way all data domain information will be encapsulated in one place.
        :param val_size:
        :return: splitting : list, length=2 * len(arrays)
        List containing train-test split of inputs.
        '''
        return train_test_split(self.X, self.y,
                                test_size=val_size,
                                stratify=self._get_stratify_strategy())

    def _get_stratify_strategy(self):
        '''
        Method to determine how stratification should work
        '''
        return self.y


class ImageDirectoryDataset(Dataset):
    '''
    Dataset to load image given a list of image name
    '''

    def __init__(self, X, y, base_dir="data", img_dir="images", transform=None):
        self.X = X
        self.y = y
        self.transform = transform
        self.base_dir = base_dir
        self.img_dir = img_dir

    def __getitem__(self, index):
        img = np.load("{}/{}/{}.npy".format(self.base_dir, self.img_dir, self.X[index]))
        img = transform.resize(img, (IMG_HEIGHT, IMG_WIDTH), preserve_range=True).astype('uint8')
        #img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.X)


if __name__ == '__main__':
    ds = MasterData()
    print(ds.train_val_split())
