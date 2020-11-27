# Dataloader for ESA's Kelvins pose estimation competition.
# https://gitlab.com/EuropeanSpaceAgency/speed-utils
import numpy as np
import json
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import torchvision.models as models

# VISUALIZATION UTILS
class Camera:
    """" Utility class for accessing camera parameters. """

    def __init__(self, img_size=(1920, 1200), input_size=(1920, 1200)):
        self.input_size = input_size
        fx = 0.0176     # focal length[m]
        fy = 0.0176     # focal length[m]
        self.nu = 1920  # number of horizontal[pixels]
        self.nv = 1200  # number of vertical[pixels]
        ppx = 5.86e-6   # horizontal pixel pitch[m / pixel]
        ppy = ppx       # vertical pixel pitch[m / pixel]
        fpx = fx / ppx  # horizontal focal length[pixels]
        fpy = fy / ppy  # vertical focal length[pixels]

        k = [[fpx,   0, self.nu/2],
             [0, fpy, self.nv/2],
             [0,   0,         1]]
        scale = np.array([[input_size[0]/img_size[0], 0, 0],
                          [0, input_size[1]/img_size[1], 0],
                          [0, 0, 1]])
        self.K = np.dot(scale, np.array(k))


def project(q, r, K, points):
    """ Projecting points to image frame to draw axes """

    # reference points in satellite frame for drawing axes
    points_body = points.T

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, points_body)

    # getting homogeneous coordinates
    points_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    points_image_plane = K.dot(points_camera_frame)

    x, y = (points_image_plane[0], points_image_plane[1])
    return x, y


def scalePoseVector(xa, ya, factor):
    # red - x, green - y, blue - z
    # input format: xa contains four x-coordinates (origin, x, y, z)
    # input format: ya contains four y-coordinates (origin, x, y, z)
    # input format: factor should be greater than zero (less than 1 for scale down)
    for i in range(1, 4):
        xa[i] = (xa[i] - xa[0]) * factor + xa[0]
        ya[i] = (ya[i] - ya[0]) * factor + ya[0]

    return xa, ya


# POSE TRANSFORMATION UTILS
def quat2dcm(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """

    # normalizing quaternion
    q = q/np.linalg.norm(q)

    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    dcm = np.zeros((3, 3))

    dcm[0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
    dcm[1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
    dcm[2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

    dcm[0, 1] = 2 * q1 * q2 + 2 * q0 * q3
    dcm[0, 2] = 2 * q1 * q3 - 2 * q0 * q2

    dcm[1, 0] = 2 * q1 * q2 - 2 * q0 * q3
    dcm[1, 2] = 2 * q2 * q3 + 2 * q0 * q1

    dcm[2, 0] = 2 * q1 * q3 + 2 * q0 * q2
    dcm[2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def q_to_mrp(q):
    return q[1:4] / (1 + q[0])


def dcm_to_q(dcm):
    q0 = -0.5*np.sqrt(dcm[0][0]+dcm[1][1]+dcm[2][2]+1)
    q1 = (dcm[1][2]-dcm[2][1])/(4*q0)
    q2 = (dcm[2][0]-dcm[0][2])/(4*q0)
    q3 = (dcm[0][1]-dcm[1][0])/(4*q0)

    return [q0, q1, q2, q3]


def mrp_to_q(sigma):
    s2 = np.dot(sigma.T, sigma)
    q0 = (1-s2)/(1+s2)
    qi = 2*sigma/(1+s2)
    return np.concatenate((np.array([q0]), qi))


def quat2dcm_batch(q):
    """ Computing direction cosine matrix from quaternion, adapted from PyNav. """
    # Ground truth quaternion will be normalized already
    bs = q.shape[0]

    dcm = torch.Tensor(np.zeros((3*bs, 3)))
    for i in range(0, bs):
        q0 = q[i].detach().cpu().numpy()[0]
        q1 = q[i].detach().cpu().numpy()[1]
        q2 = q[i].detach().cpu().numpy()[2]
        q3 = q[i].detach().cpu().numpy()[3]

        dcm[3*i+0, 0] = 2 * q0 ** 2 - 1 + 2 * q1 ** 2
        dcm[3*i+1, 1] = 2 * q0 ** 2 - 1 + 2 * q2 ** 2
        dcm[3*i+2, 2] = 2 * q0 ** 2 - 1 + 2 * q3 ** 2

        dcm[3*i+0, 1] = 2 * q1 * q2 + 2 * q0 * q3
        dcm[3*i+0, 2] = 2 * q1 * q3 - 2 * q0 * q2

        dcm[3*i+1, 0] = 2 * q1 * q2 - 2 * q0 * q3
        dcm[3*i+1, 2] = 2 * q2 * q3 + 2 * q0 * q1

        dcm[3*i+2, 0] = 2 * q1 * q3 + 2 * q0 * q2
        dcm[3*i+2, 1] = 2 * q2 * q3 - 2 * q0 * q1

    return dcm


def dcm_to_euler321_batch(dcm):
    bs = int(dcm.shape[0]/3)

    e = torch.Tensor(np.zeros((bs, 3)))
    for i in range(0,bs):
        e[i][1] = np.rad2deg(-np.arcsin(dcm[3*i+0][2]))
        if e[i][1] < 0: e[i][1] = 2*np.pi - e[i][1]
            
        e[i][0] = np.rad2deg(np.arctan2(dcm[3*i+0][1],dcm[3*i+0][0]))
        if e[i][0] < 0: e[i][0] = 2*np.pi - e[i][0]
            
        e[i][2] = np.rad2deg(np.arctan2(dcm[3*i+1][2],dcm[3*i+2][2]))
        if e[i][2] < 0: e[i][2] = 2*np.pi - e[i][2]

    return e


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


def origin_reg_conversion(label, K, t_out):
    u = (label[4:11][0] + label[4:11][2]) / 2.0
    v = (label[4:11][1] + label[4:11][3]) / 2.0
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    tx = (u + t_out[0] - cx*t_out[2])/fx
    ty = (v + t_out[1] - cy*t_out[2])/fy
#     tx = t_out[2]/fx*(u + t_out[0]-cx)
#     ty = t_out[2]/fy*(v + t_out[1]-cy)
    return np.array([tx, ty, t_out[2]])

# DATASET UTILS
class SpeedDataset(Dataset):

    """ SPEED dataset that can be used with DataLoader for PyTorch training. """

    def __init__(self, split='train', speed_root='', annotations_root='', input_size=(256, 256), split_index=None, transform=None, sanity_check=None):

        if split not in {'train', 'test', 'real_test'}:
            raise ValueError(
                'Invalid split, has to be either \'train\', \'test\' or \'real_test\'')

        with open(os.path.join(annotations_root, split + '.json'), 'r') as f:
            label_list = json.load(f)

        self.sample_ids = [label['filename'] for label in label_list]

        self.train = split == 'train'
        split_idx = -1 if split_index is None else split_index

        if self.train:
            self.labels = [{'q': label['q_vbs2tango'], 'r': label['r_Vo2To_vbs_true'],
                            'bbox': label['bbox'], 'wireframe': label['wireframe']} for label in label_list]

            if sanity_check is not None:  # to overfit network on one training image
                self.sample_ids = [self.sample_ids[sanity_check]]
                self.labels = [self.labels[sanity_check]]
            else:
                self.sample_ids = self.sample_ids[:split_idx]
                self.labels = self.labels[:split_idx]

        self.image_root = os.path.join(speed_root, 'images', split)

        self.transform = transform
        self.wireframe_vertices = np.array([[0.37, 0.285, 0, 1], [-0.37, 0.285, 0, 1], [-0.37, -0.285, 0, 1], [0.37, -0.285, 0, 1],
                                            [0.37, 0.285, 0.295, 1], [-0.37, 0.285, 0.295, 1], [-0.37, -0.285, 0.295, 1], [0.37, -0.285, 0.295, 1]])
        self.axes_vertices = np.array(
            [[0, 0, 0, 1], [1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1]])

        if self.transform is not None:
            self.input_size = self.transform.transforms[0].size
        else:
            self.input_size = input_size

    def __len__(self):
        return len(self.sample_ids)

    def __getitem__(self, idx):
        # sample_id = self.sample_ids[idx]
        img_name = os.path.join(self.image_root, self.sample_ids[idx])

        # note: despite grayscale images, we are converting to 3 channels here,
        # since most pre-trained networks expect 3 channel input
        pil_image = Image.open(img_name).convert('RGB')

        if self.train:
            q, r, bbox, wireframe = self.labels[idx]['q'], self.labels[idx][
                'r'], self.labels[idx]['bbox'], self.labels[idx]['wireframe']
            y = np.concatenate([q, r, bbox, wireframe])
        else:
            y = self.sample_ids[idx]

        if self.transform is not None:
            torch_image, y = self.transform(pil_image, y)
            # Retrieve modified camera intrinsics as a result of transformation
            cP = (self.transform.transforms[0].cx,
                  self.transform.transforms[0].cy)
            cropSize = self.transform.transforms[0].cropSize
            camera = Camera(img_size=(cropSize, cropSize),
                            input_size=self.input_size)
            camera.K[0][2] = (camera.nu/2 + cropSize/2 - cP[0]
                              ) * camera.input_size[0] / cropSize
            camera.K[1][2] = (camera.nv/2 + cropSize/2 - cP[1]
                              ) * camera.input_size[1] / cropSize
            K = camera.K
        else:
            torch_image = pil_image
            camera = Camera()
            K = camera.K

        return torch_image, y, K

    def visualize(self, img, target, K, factor=1, ax=None, bbox=False, dbox=False):
        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()

        ax.imshow(img)

        target_q = target[:4]
        target_r = target[4:7]
        xa, ya = project(target_q, target_r, K, self.axes_vertices)
        xa, ya = scalePoseVector(xa, ya, factor)
        ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] -
                 ya[0], head_width=10, color='r')
        ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] -
                 ya[0], head_width=10, color='g')
        ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] -
                 ya[0], head_width=10, color='b')
        origin = [xa[0], ya[0]]

        if bbox:
            xa, ya = project(target_q, target_r, K, self.wireframe_vertices)
            ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[1], ya[1], xa[2] - xa[1], ya[2] - ya[1],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[2], ya[2], xa[3] - xa[2], ya[3] - ya[2],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[3], ya[3], xa[0] - xa[3], ya[0] - ya[3],
                     head_width=None, head_length=None, color='lime')

            ax.arrow(xa[4], ya[4], xa[5] - xa[4], ya[5] - ya[4],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[5], ya[5], xa[6] - xa[5], ya[6] - ya[5],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[6], ya[6], xa[7] - xa[6], ya[7] - ya[6],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[7], ya[7], xa[4] - xa[7], ya[4] - ya[7],
                     head_width=None, head_length=None, color='lime')

            ax.arrow(xa[0], ya[0], xa[4] - xa[0], ya[4] - ya[0],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[1], ya[1], xa[5] - xa[1], ya[5] - ya[1],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[2], ya[2], xa[6] - xa[2], ya[6] - ya[2],
                     head_width=None, head_length=None, color='lime')
            ax.arrow(xa[3], ya[3], xa[7] - xa[3], ya[7] - ya[3],
                     head_width=None, head_length=None, color='lime')

        if dbox:
            x_min, y_min, x_max, y_max = target[7:11]
            x_c = (x_min + x_max)/2
            y_c = (y_min + y_max)/2

            ax.arrow(x_min, y_min, x_max-x_min, 0, head_width=None,
                     head_length=None, color='lime')
            ax.arrow(x_max, y_min, 0, y_max-y_min, head_width=None,
                     head_length=None, color='lime')
            ax.arrow(x_max, y_max, x_min-x_max, 0, head_width=None,
                     head_length=None, color='lime')
            ax.arrow(x_min, y_max, 0, y_min-y_max, head_width=None,
                     head_length=None, color='lime')
        return origin


def process_json_dataset(root_dir):
    with open(os.path.join(root_dir, 'train.json'), 'r') as f:
        train_images_labels = json.load(f)

    with open(os.path.join(root_dir, 'test.json'), 'r') as f:
        test_image_list = json.load(f)

    with open(os.path.join(root_dir, 'real_test.json'), 'r') as f:
        real_test_image_list = json.load(f)

    partitions = {'test': [], 'train': [], 'real_test': []}
    labels = {}

    for image_ann in train_images_labels:
        partitions['train'].append(image_ann['filename'])
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'],
                                         'r': image_ann['r_Vo2To_vbs_true'],
                                         'bbox': image_ann['bbox'],
                                         'wireframe': image_ann['wireframe']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels
