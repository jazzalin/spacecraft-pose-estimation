# Starter kit for ESA's Kelvins pose estimation competition.
# https://gitlab.com/EuropeanSpaceAgency/speed-utils
import numpy as np
import json
import os
from PIL import Image, ImageDraw
from matplotlib import pyplot as plt


class Camera:

    """" Utility class for accessing camera parameters. """

    fx = 0.0176  # focal length[m]
    fy = 0.0176  # focal length[m]
    nu = 1920  # number of horizontal[pixels]
    nv = 1200  # number of vertical[pixels]
    ppx = 5.86e-6  # horizontal pixel pitch[m / pixel]
    ppy = ppx  # vertical pixel pitch[m / pixel]
    fpx = fx / ppx  # horizontal focal length[pixels]
    fpy = fy / ppy  # vertical focal length[pixels]
    k = [[fpx,   0, nu / 2],
         [0,   fpy, nv / 2],
         [0,     0,      1]]
    K = np.array(k)


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
        labels[image_ann['filename']] = {'q': image_ann['q_vbs2tango'], 'r': image_ann['r_Vo2To_vbs_true']}

    for image in test_image_list:
        partitions['test'].append(image['filename'])

    for image in real_test_image_list:
        partitions['real_test'].append(image['filename'])

    return partitions, labels


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


def project(q, r):

        """ Projecting points to image frame to draw axes """

        # reference points in satellite frame for drawing axes
        p_axes = np.array([[0, 0, 0, 1],
                           [1, 0, 0, 1],
                           [0, 1, 0, 1],
                           [0, 0, 1, 1]])
        points_body = np.transpose(p_axes)

        # transformation to camera frame
        pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
        p_cam = np.dot(pose_mat, points_body)

        # getting homogeneous coordinates
        points_camera_frame = p_cam / p_cam[2]

        # projection to image plane
        points_image_plane = Camera.K.dot(points_camera_frame)

        x, y = (points_image_plane[0], points_image_plane[1])
        return x, y


def project_keypoints(q, r, keypoints):
    """ Projecting TANGO 3D vertices to image frame to create wireframe """
    keypoints_body = np.transpose(keypoints)

    # transformation to camera frame
    pose_mat = np.hstack((np.transpose(quat2dcm(q)), np.expand_dims(r, 1)))
    p_cam = np.dot(pose_mat, keypoints_body)

    # getting homogeneous coordinates
    keypoints_camera_frame = p_cam / p_cam[2]

    # projection to image plane
    keypoints_image_plane = Camera.K.dot(keypoints_camera_frame)

    x, y = (keypoints_image_plane[0], keypoints_image_plane[1])
    return x, y


class SatellitePoseEstimationDataset:

    """ Class for dataset inspection: easily accessing single images, and corresponding ground truth pose data. """

    def __init__(self, root_dir='/datasets/speed_debug', annotation_dir='../annotations/'):
        self.partitions, self.labels = process_json_dataset(root_dir)
        self.root_dir = root_dir
        self.annotations_dir = annotation_dir
        # 8 TANGO vertices in body frame (homogeneous coordinates): [A, B, C, D, E, F, G, H]
        self.wireframe_vertices = np.array([[0.37, 0.285, 0, 1],
                                            [-0.37, 0.285, 0, 1],
                                            [-0.37, -0.285, 0, 1],
                                            [0.37, -0.285, 0, 1],
                                            [0.37, 0.285, 0.295, 1],
                                            [-0.37, 0.285, 0.295, 1],
                                            [-0.37, -0.285, 0.295, 1],
                                            [0.37, -0.285, 0.295, 1]]) 

        # Number of images within each subfolder of SPEED dataset
        self.n_train = 12000
        self.n_test = 2998
        self.n_real = 5
        self.n_real_test = 300

    def get_image(self, i=0, split='train'):

        """ Loading image as PIL image. """

        img_name = self.partitions[split][i]
        img_name = os.path.join(self.root_dir, 'images', split, img_name)
        image = Image.open(img_name).convert('RGB')
        return image

    def get_pose(self, i=0):

        """ Getting pose label for image. """

        img_id = self.partitions['train'][i]
        q, r = self.labels[img_id]['q'], self.labels[img_id]['r']
        return q, r

    def get_label(self, i=0, split='train'):

        """ Getting annotation for image """

        img_id = self.partitions[split][i]
        return self.labels[img_id]
    

    def visualize(self, i, partition='train', ax=None, pose=True, bb=False, db=False):

        """ Visualizing image, with ground truth pose with axes projected to training image. """

        if ax is None:
            ax = plt.gca()
        img = self.get_image(i)
        ax.imshow(img)

        # no pose label for test
        if partition == 'train':
            if pose:
                q, r = self.get_pose(i)
                xa, ya = project(q, r) # NOTE: first coordinates are those of the interface point (body frame origin)
                ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=30, color='r')
                ax.arrow(xa[0], ya[0], xa[2] - xa[0], ya[2] - ya[0], head_width=30, color='g')
                ax.arrow(xa[0], ya[0], xa[3] - xa[0], ya[3] - ya[0], head_width=30, color='b')
            if bb or db:
                q, r = self.get_pose(i)
                xa, ya = project_keypoints(q, r, self.wireframe_vertices)
                if bb:
                    ax.arrow(xa[0], ya[0], xa[1] - xa[0], ya[1] - ya[0], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[1], ya[1], xa[2] - xa[1], ya[2] - ya[1], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[2], ya[2], xa[3] - xa[2], ya[3] - ya[2], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[3], ya[3], xa[0] - xa[3], ya[0] - ya[3], head_width=None, head_length=None, color='lime')

                    ax.arrow(xa[4], ya[4], xa[5] - xa[4], ya[5] - ya[4], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[5], ya[5], xa[6] - xa[5], ya[6] - ya[5], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[6], ya[6], xa[7] - xa[6], ya[7] - ya[6], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[7], ya[7], xa[4] - xa[7], ya[4] - ya[7], head_width=None, head_length=None, color='lime')

                    ax.arrow(xa[0], ya[0], xa[4] - xa[0], ya[4] - ya[0], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[1], ya[1], xa[5] - xa[1], ya[5] - ya[1], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[2], ya[2], xa[6] - xa[2], ya[6] - ya[2], head_width=None, head_length=None, color='lime')
                    ax.arrow(xa[3], ya[3], xa[7] - xa[3], ya[7] - ya[3], head_width=None, head_length=None, color='lime')
                if db:
                    x_min = np.min(xa)
                    x_max = np.max(xa)
                    y_min = np.min(ya)
                    y_max = np.max(ya)
                    
                    ax.arrow(x_min, y_min, x_max-x_min, 0, head_width=None, head_length=None, color='lime')
                    ax.arrow(x_max, y_min, 0, y_max-y_min, head_width=None, head_length=None, color='lime')
                    ax.arrow(x_max, y_max, x_min-x_max, 0, head_width=None, head_length=None, color='lime')
                    ax.arrow(x_min, y_max, 0, y_min-y_max, head_width=None, head_length=None, color='lime')
        return

            
    def annotate(self, partition='train'):

        """ Annotating images with wireframe and detection boxes (bbox) ground truths """

        # Read original labels from file
        original = "original_" + partition + ".json"
        with open(os.path.join(self.annotations_dir, original), 'r') as f:
            image_labels = json.load(f)

        # Output updated labels to new annotation file
        new = partition + ".json"

        for i, img in enumerate(image_labels):
            q, r = self.get_pose(i)
            xa, ya = project_keypoints(q, r, self.wireframe_vertices)
            x_min = np.min(xa)
            x_max = np.max(xa)
            y_min = np.min(ya)
            y_max = np.max(ya)

            img['bbox'] = [x_min, y_min, x_max, y_max]
            img['wireframe'] = [xa[0], ya[0], xa[1], ya[1], xa[2], ya[2], xa[3], ya[3], xa[4], ya[4], xa[5], ya[5], xa[6], ya[6], xa[7], ya[7]]

        new_labels_json = json.dumps(image_labels, indent = 4) 
        with open(os.path.join(self.annotations_dir, new), 'w') as out:
            out.write(new_labels_json)

        return

    def preprocess(self, partition='train', ax=None):
        """ Preprocessing images and saving with wireframe """

        if partition == 'train':
            for i in range(self.n_train):
                img = self.get_image(i)
                img_name = self.partitions[partition][i]
                preprocess_filename = '../speed_wireframe/' + partition + '/' + img_name
                img1 = ImageDraw.Draw(img)
                
                q, r = self.get_pose(i)
                xa, ya = project_keypoints(q, r, self.wireframe_vertices)
                shape = [(xa[0], ya[0]), (xa[1], ya[1]), (xa[2], ya[2]), (xa[3], ya[3]), (xa[0], ya[0]), (xa[4], ya[4])]
                img1.line(shape, fill="lime", width=0)
                shape = [(xa[5], ya[5]), (xa[6], ya[6]), (xa[7], ya[7]), (xa[4], ya[4]), (xa[5], ya[5]), (xa[1], ya[1])]
                img1.line(shape, fill="lime", width=0)
                shape = [(xa[6], ya[6]), (xa[2], ya[2])]
                img1.line(shape, fill="lime", width=0)
                shape = [(xa[7], ya[7]), (xa[3], ya[3])]
                img1.line(shape, fill="lime", width=0)
                img.save(preprocess_filename)
        return

