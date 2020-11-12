# Test quaternion to Euler 3-2-1
# 
import numpy as np
import json
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt
from random import randint
from dataloader_utils import *
torch.manual_seed(42)
from matplotlib import cm
import torchvision.transforms.functional as tF

class JointRescrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        target_q = target[:4]
        target_r = target[4:7]
        target_db = target[7:11] 
        target_bb = target[11:]

        sz = self.size[0]
        # Determine crop size that will still capture entire satellite
        x = [target_bb[i] for i in range(0, len(target_bb), 2)]
        y = [target_bb[i] for i in range(1, len(target_bb), 2)]
        xmin = min(x); xmax = max(x)
        ymin = min(y); ymax = max(y)
        fS = 1.75
        dx = xmax - xmin
        dy = ymax - ymin
        self.cropSize = int(fS * max(dx, dy))
        if self.cropSize > min(img.size):
            self.cropSize = min(img.size)
#         print("cropSize: ", cropSize)
        
        # Crop Image
        self.cx = sum(x)/len(x)
        self.cy = sum(y)/len(y)
        top = self.cy - self.cropSize/2
        left = self.cx - self.cropSize/2
        img = tF.crop(img, top, left, self.cropSize, self.cropSize)
        
        # Crop translation of points
        xtarget = [self.cropSize/2 + (i - self.cx) for i in x]
        ytarget = [self.cropSize/2 + (i - self.cy) for i in y]
        target_bb = [None]*(len(xtarget)+len(ytarget))
        target_bb[::2] = xtarget
        target_bb[1::2] = ytarget
        target_bb = np.array(target_bb)
        
        # Rescale to input size
        norm = img.size[1]
        img = tF.resize(img, sz)
        target_bb = [sz * i / norm for i in target_bb]
        target = np.concatenate((target_q, target_r, target_db, target_bb))

        return img, target

class JointToTensor(object):
    def __call__(self, img, target):
        return tF.to_tensor(img), torch.from_numpy(target)

class JointCompose(object):
    def __init__(self, transforms):
        """
        params: 
           transforms (list) : list of transforms
        """
        self.transforms = transforms

    # We override the __call__ function such that this class can be
    # called as a function i.e. JointCompose(transforms)(img, target)
    # Such classes are known as "functors"
    def __call__(self, img, target):
        """
        params:
            img (PIL.Image)    : input image
            target (PIL.Image) : ground truth label 
        """
        for t in self.transforms:
            img, target = t(img, target)
        return img, target

def dcm_to_euler321(dcm):
    th_2 = -np.arcsin(dcm[0][2])
    # th_1 = np.arccos(dcm[0][0]/np.cos(th_2))
    # th_3 = np.arcsin(dcm[1][2]/np.cos(th_2))
    th_1 = np.arctan2(dcm[0][1],dcm[0][0])
    th_3 = np.arctan2(dcm[1][2],dcm[2][2])

    return th_1, th_2, th_3

def euler321_to_dcm(th_1, th_2, th_3):
    dcm = np.zeros((3, 3))
    dcm[0][0] = np.cos(th_1)*np.cos(th_2)
    dcm[0][1] = np.sin(th_1)*np.cos(th_2)
    dcm[0][2] = -np.sin(th_2)
    dcm[1][0] = -np.sin(th_1)*np.cos(th_3) + np.cos(th_1)*np.sin(th_2)*np.sin(th_3)
    dcm[1][1] = np.cos(th_1)*np.cos(th_3) + np.sin(th_1)*np.sin(th_2)*np.sin(th_3)
    dcm[1][2] = np.cos(th_2)*np.sin(th_3)
    dcm[2][0] = np.sin(th_1)*np.sin(th_3) + np.cos(th_1)*np.sin(th_2)*np.cos(th_3)
    dcm[2][1] = -np.cos(th_1)*np.sin(th_3) + np.sin(th_1)*np.sin(th_2)*np.cos(th_3)
    dcm[2][2] = np.cos(th_2)*np.cos(th_3)

    return dcm

def dcm_to_q(dcm):
    q0 = -0.5*np.sqrt(dcm[0][0]+dcm[1][1]+dcm[2][2]+1)
    q1 = (dcm[1][2]-dcm[2][1])/(4*q0)
    q2 = (dcm[2][0]-dcm[0][2])/(4*q0)
    q3 = (dcm[0][1]-dcm[1][0])/(4*q0)

    return [q0, q1, q2, q3]


def main():
    dataset_root_dir = '../../speed'
    annotations_root_dir = '../annotations'
    sample_transform = transforms.Compose([
        transforms.RandomRotation((-30, 30))
    ])
    sample_dataset = SpeedDataset(speed_root=dataset_root_dir, annotations_root=annotations_root_dir, transform=None)
    
    # Transforms
    train_transform = JointCompose([
        JointRescrop((256,256)),
        JointToTensor(),
    ])

    sanity_transform = JointCompose([
        JointRescrop((256,256)),
        JointToTensor(),
    ])

    val_transform = JointCompose([
        JointRescrop((256,256)),
        JointToTensor(),
    ])

    # Datasets
    # NOTE: we don't have the labels for the test set, so we need to split the training set
    training_dataset = SpeedDataset(
        split="train",
        split_index=100, # used to make a smaller training set for dev
        speed_root=dataset_root_dir,
        annotations_root=annotations_root_dir,        
        transform=train_transform
    )

    sanity_dataset = SpeedDataset(
        split="train",
        sanity_check=100,
        speed_root=dataset_root_dir,
        annotations_root=annotations_root_dir,        
        transform=sanity_transform
    )

    # TEST
    sample, target = training_dataset[4]
    image1, label1 = JointRescrop((256, 256))(tF.to_pil_image(sample), target.numpy())
    training_dataset.visualize(image1, label1, factor=0.6, bbox=True)
    
    target_q = label1[:4]
    target_r = label1[4:7]

    print("q: ", target_q)
    print("r: ", target_r)

    dcm = quat2dcm(target_q)
    print("dcm:", dcm)

    theta_1, theta_2, theta_3 = dcm_to_euler321(dcm)

    print("th1: ", np.rad2deg(theta_1))
    print("th2: ", np.rad2deg(theta_2))
    print("th3: ", np.rad2deg(theta_3))

    dcm = euler321_to_dcm(theta_1, theta_2, theta_3)
    print("dcm: ", dcm)

    q = dcm_to_q(dcm)
    print("q: ", q)

    # plt.show()


if __name__ == "__main__":
    main()       