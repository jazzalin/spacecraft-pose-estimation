from submission.submission import SubmissionWriter
from model.model import MangoNet, TangoNet
from model.dataloader_utils import SpeedDataset
from model.data_transforms import *
from train import *
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import json
import os

""" Valid submission generation example. """

# load test image list
dataset_root = '../speed'
annotations_root = './annotations'
with open(os.path.join(dataset_root, 'test.json'), 'r') as f:
    test_images = json.load(f)
with open(os.path.join(dataset_root, 'real_test.json'), 'r') as f:
    real_test_images = json.load(f)

submission = SubmissionWriter()
val_net = TangoNet()
val_net.load_state_dict(torch.load("best_prv_5000_2.pth"))
val_net.to(device)
val_net.eval()

transform = JointCompose([
    JointRescropTest((256,256)),
    JointToTensor(),
    # JointNormalize([0.5], [0.5])
])

test_dataset = SpeedDataset(
    split="test",
    speed_root=dataset_root,
    annotations_root=annotations_root,        
    transform=transform
)

real_test_dataset = SpeedDataset(
    split="real_test",
    speed_root=dataset_root,
    annotations_root=annotations_root,        
    transform=transform
)

test_loader = DataLoader(test_dataset, batch_size=1, num_workers=6, shuffle=False)
real_test_loader = DataLoader(real_test_dataset, batch_size=1, num_workers=6, shuffle=False)

# iterating over all test and real test images, appending submission
for i, data in enumerate(test_loader):
    filename = test_loader.dataset.sample_ids[i]
    img, target, K = data
    img = img.cuda()

    t, att = val_net.forward(img)

    # Conversion: prv --> q
    R_pred = torch.matrix_exp(skew_symmetric(att))
    q = dcm_to_ep(R_pred)
    q = q.detach().cpu().numpy()
    if np.any(np.isnan(q)):
        print("Test: Nans found in q!")
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        print(q[0])

    # Conversion: [delta u, delta v, tz] -> r = [tx, ty, tz]
    r = origin_reg_conversion(target.detach().cpu().numpy()[0], K.detach().cpu().numpy()[0], t.detach().cpu().numpy()[0])
    if np.any(np.isnan(r)):
        print("Test: Nans found in r!")
    submission.append_test(filename, q[0], r.tolist())

for i, data in enumerate(real_test_loader):
    filename = real_test_loader.dataset.sample_ids[i]
    img, target, K = data
    img = img.cuda()

    t, att = val_net.forward(img)

    # Conversion: prv --> q
    R_pred = torch.matrix_exp(skew_symmetric(att))
    q = dcm_to_ep(R_pred)
    q = q.detach().cpu().numpy()
    if np.any(np.isnan(q)):
        print("Real_test: Nans found in q!")
        q = np.array([[1.0, 0.0, 0.0, 0.0]])
        print(q[0])

    # Conversion: [delta u, delta v, tz] -> r = [tx, ty, tz]
    r = origin_reg_conversion(target.detach().cpu().numpy()[0], K.detach().cpu().numpy()[0], t.detach().cpu().numpy()[0])
    if np.any(np.isnan(r)):
        print("Real_test: Nans found in r!")
    submission.append_real_test(filename, q[0], r.tolist())

submission.export(suffix='prv')
print('Submission exported.')
