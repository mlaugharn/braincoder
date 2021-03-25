import torch
import numpy as np
import pandas as pd
import os
from pathlib import Path
import imageio
from torchvision.datasets.imagenet import ImageNet
import config
from tqdm.autonotebook import tqdm
from PIL import Image
from torch import nn
import einops
from einops.layers.torch import Rearrange
from torchvision import transforms as T
import torchvision

def image_prepare(img,size):
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    t = T.Compose([T.ToTensor(), normalize]) if config.normalize else T.Compose([T.ToTensor()])
    out_img = np.zeros([size,size,3])
    s = img.shape
    r = s[0]
    c = s[1]

    trimSize = np.min([r, c])
    lr = int((c - trimSize) / 2)
    ud = int((r - trimSize) / 2)
    img = img[ud:min([(trimSize + 1), r - ud]) + ud, lr:min([(trimSize + 1), c - lr]) + lr]

    # img = imresize(img, size=[size, size], interp=interpolation)
    img = np.array(Image.fromarray(img).resize(size=[size, size], resample=Image.CUBIC))
    img = t(img).permute(1,2,0).numpy()
    if (np.ndim(img) == 3):
        out_img = img
    else:
        out_img[ :, :, 0] = img
        out_img[ :, :, 1] = img
        out_img[ :, :, 2] = img

    return out_img#/255.0

def gen_npz(test_csv='images/image_test_id.csv',
            train_csv='images/image_training_id.csv', 
            size=config.img_size,
            interpolation='cubic'):
    test_im = pd.read_csv(test_csv, header=None)
    train_im = pd.read_csv(train_csv, header=None)
    
    test_images = np.zeros([50, size, size, 3])
    train_images = np.zeros([1200, size, size, 3])
    
    for idx, file in tqdm(enumerate(test_im[1])):
        img = imageio.imread(Path('images/test')/file)
        test_images[idx] = image_prepare(img, size)
    
    for idx, file in tqdm(enumerate(train_im[1])):
        try:
            img = imageio.imread(Path('images/training')/file)
            train_images[idx] = image_prepare(img, size)
        except:
            print(file)
    
    np.savez(config.npz_file, train_images=train_images, test_images=test_images)

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, kind, labels, mris, images=None,):
        self.kind = kind # either 'train_images' or 'test_images'
        if images is not None:
            self.images=images
        else:
            assert os.path.exists(config.npz_file), "Run gen_npz first"
            self.images = np.load(config.npz_file)[self.kind]
#             if self.kind == 'test_images': self.images = self.images[labels]
        self.images = self.images#.transpose(0, 3, 1, 2)
        self.mris = mris
        self.labels = labels
    
    def __len__(self):
        return self.images.shape[0]
    
    def __getitem__(self, idx):
        if self.kind == 'train_images':
            return self.images[idx], self.mris[idx], idx, 
            # return self.images[self.labels[idx]], self.labels[idx]
        elif self.kind == 'test_images':
            # from 1750 to 50
            return self.images[idx], self.mris[idx], idx
            # return self.images[self.labels[idx]], self.labels[idx]
        
# strongly influenced (ie, basically lifted entirely from) kamitani_data_handler.py in ssfrmi2im

from scipy.io import loadmat
import numpy as np
import pandas as pd
import sklearn.preprocessing
from sklearn import preprocessing

class kamitani_data_handler():
    def __init__(self,
                 matlab_file,
                 test_img_csv = 'KamitaniData/imageID_test.csv',
                train_img_csv='KamitaniData/imageID_training.csv',
                voxel_spacing=3,
                log=0):
        mat = loadmat(matlab_file)
        self.data = mat['dataSet'][:,3:]
        self.sample_meta = mat['dataSet'][:,:3]
        meta = mat['metaData']
        
        self.meta_keys = list(l[0] for l in meta[0][0][0][0])
        self.meta_desc = list(l[0] for l in meta[0][0][1][0])
        self.voxel_meta = np.nan_to_num(meta[0][0][2][:,3:])
        test_img_df = pd.read_csv(test_img_csv, header=None)
        train_img_df = pd.read_csv(train_img_csv, header=None)
        self.test_img_id=test_img_df[0].values
        self.train_img_id=train_img_df[0].values
        self.sample_type = {'train': 1, 'test': 2, 'test_imagine': 3}
        self.voxel_spacing = voxel_spacing
        self.log = log
        
    def get_meta_field(self, field = 'DataType'):
        index = self.meta_keys.index(field)
        if (index < 3): # 3 first keys are sample meta
            return self.sample_meta[:, index]
        else:
            return self.voxel_meta[index]
        
    def print_meta_desc(self):
        print(self.meta_desc)
    
    def get_labels(self, imag_data = 0, test_run_list = None):
        le = preprocessing.LabelEncoder()
        
        img_ids = self.get_meta_field('Label')
        type = self.get_meta_field('DataType')
        train = (type == self.sample_type['train'])
        test = (type == self.sample_type['test'])
        imag = (type == self.sample_type['test_imagine'])
        
        img_ids_train = img_ids[train]
        img_ids_test = img_ids[test]
        img_ids_imag = img_ids[imag]
        
        train_labels = []
        test_labels = []
        imag_labels = []
        
        for id in img_ids_test:
            idx = (np.abs(id - self.test_img_id)).argmin()
            test_labels.append(idx)
        
        for id in img_ids_train:
            idx = (np.abs(id - self.train_img_id)).argmin()
            train_labels.append(idx)
            
        for id in img_ids_imag:
            idx = (np.abs(id - self.test_img_id)).argmin()
            imag_labels.append(idx)
        
        if (test_run_list is not None):
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]
            
            select = np.in1d(run, test_run_list)
            test_labels = test_labels[select]
            
        # imag_labels = le.fit_transform(img_ids_imag)
        if(imag_data):
            return np.array(train_labels), np.array(test_labels), np.array(imag_labels)
        else:
            return np.array(train_labels), np.array(test_labels)
        
    def get_data(self,
                 normalize = 1,
                roi = 'ROI_VC',
                imag_data = 0,
                test_run_list = None): # normalize 0- no, 1- per run, 2- train/test separately
        type = self.get_meta_field('DataType')
        train = (type == self.sample_type['train'])
        test = (type == self.sample_type['test'])
        test_imag = (type == self.sample_type['test_imagine'])
        test_all = np.logical_or(test, test_imag)
        
        roi_select = self.get_meta_field(roi).astype(bool)
        data = self.data[:, roi_select]
        
        if self.log == 1:
            data = np.log(1 + np.abs(data)) * np.sign(data)
        
        if normalize == 1:
            run = self.get_meta_field('Run').astype(int)-1
            num_runs = np.max(run) + 1
            data_norm = np.zeros(data.shape)
            
            for r in range(num_runs):
                data_norm[r==run] = sklearn.preprocessing.scale(data[r==run])
            train_data = data_norm[train]
            test_data = data_norm[test]
            test_all = data_norm[test_all]
            test_imag = data_norm[test_imag]
        
        else:
            train_data = data[train]
            test_data = data[test]
            if normalize == 2:
                train_data = sklearn.preprocessing.scale(train_data)
                test_data = sklearn.preprocessing.scale(test_data)
                
        if self.log==2:
            train_data = np.log(1+np.abs(train_data)) * np.sign(train_data)
            test_data = np.log(1+np.abs(test_data)) * np.sign(test_data)
            train_data = sklearn.preprocessing.scale(train_data)
            test_data = sklearn.preprocessing.scale(test_data)
            
        test_labels = self.get_labels()[1]
        imag_labels = self.get_labels(1)[2]
        num_labels = max(test_labels)+1
        
        test_data_avg = np.zeros([num_labels, test_data.shape[1]])
        test_imag_avg = np.zeros([num_labels, test_data.shape[1]])
        
        if test_run_list is not None:
            run = self.get_meta_field('Run')
            test = (self.get_meta_field('DataType') == 2).astype(bool)
            run = run[test]
            
            select = np.in1d(run, test_run_list)
            test_data = test_data[select, :]
            test_labels = test_labels[select]
            
        for i in range(num_labels):
            test_data_avg[i] = np.mean(test_data[test_labels==i], axis=0)
            test_imag_avg[i] = np.mean(test_imag[imag_labels==i], axis=0)
            
        if imag_data:
            return train_data, test_data, test_data_avg, test_imag, test_imag_avg
        else:
            return train_data, test_data, test_data_avg
        
    def get_voxel_loc(self):
        x = self.get_meta_field('voxel_x')
        y = self.get_meta_field('voxel_y')
        z = self.get_meta_field('voxel_z')
        dim = [int(x.max() - x.min()+1), int(y.max()-y.min()+1), int(z.max()-z.min()+1)]
        return [x,y,z], dim
            
def prepare_setup(batch_size=config.batch_size):
    kamitani_data_mat = './prj/data/SupplementaryFiles/Subject3.mat'
    handler = kamitani_data_handler(kamitani_data_mat,
                               test_img_csv = './prj/data/SupplementaryFiles/imageID_test.csv',
                               train_img_csv='./prj/data/SupplementaryFiles/imageID_training.csv')
    Y,Y_test,Y_test_avg = handler.get_data(roi='ROI_VC')
    labels_train, labels = handler.get_labels()
    NUM_VOXELS = Y.shape[1]
    images = np.load(config.npz_file)
    X = images['train_images']
    X = X[labels_train] # get images in order of kamitani labels
    X_test = images['test_images']
    # X_test = X_test[labels]

    print(X.shape, X_test.shape, labels_train.shape, labels.shape)
    
    train_ds = ImageDataset(kind='train_images', labels=labels_train, mris = Y, images=X)
    train_dl = torch.utils.data.DataLoader(train_ds, pin_memory=True, batch_size=batch_size, shuffle=True)
    test_ds = ImageDataset(kind='test_images', labels=labels, mris = Y_test_avg, images=X_test)
    test_dl = torch.utils.data.DataLoader(test_ds, pin_memory=True, batch_size=batch_size, shuffle=True)
    return handler, Y, Y_test, Y_test_avg, labels_train, labels, NUM_VOXELS, images, X, X_test, train_dl, test_dl

class ImageNetDataset(torch.utils.data.Dataset):
    def __init__(self, transform):
        self.main_dir = 'val/'
        self.transform = transform
        self.total_imgs = os.listdir(self.main_dir)
        self.cache = {}

    def __len__(self):
        return len(self.total_imgs)

    def __getitem__(self, idx):
        idx = idx % 50000
        if idx not in self.cache:
            try:
                img_loc = os.path.join(self.main_dir, self.total_imgs[idx])
                image = Image.open(img_loc).convert("RGB")
                tensor_image = self.transform(image)#.permute(1, 2, 0)

                self.cache[idx] = tensor_image
            except OSError as e:
                print(f"problem loading {idx}: {e}")
                print(f"reusing prior index image for {idx}")
                self.cache[idx] = self.cache[idx-1]
        return self.cache[idx]

def imagenet_dl():
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    t = T.Compose([T.Resize((config.img_size, config.img_size)), T.ToTensor(), normalize]) if config.normalize else T.Compose([T.Resize((config.img_size, config.img_size)), T.ToTensor()])
    ds = ImageNetDataset(t)
    return torch.utils.data.DataLoader(ds,
                    pin_memory=True,
                    batch_size=config.batch_size,
                    shuffle=True
                    )

class TestFMRIDataset(torch.utils.data.Dataset):
    def __init__(self, Y_test_avg):
        self.fmris = Y_test_avg
    def __len__(self):
        return len(self.fmris)
    def __getitem__(self, idx):
        return self.fmris[idx]

def make_test_fmri_dl(Y_test_avg):
    ds = TestFMRIDataset(Y_test_avg)
    return torch.utils.data.DataLoader(ds,
                                    # pin_memory=True,
                                    batch_size=config.batch_size,
                                    shuffle=True)
    