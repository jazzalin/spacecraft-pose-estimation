from submission import SubmissionWriter
from submission import SpeedDataset
from model.data_transforms import *
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
val_net.load_state_dict(torch.load("best_model.pth"))
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
# for image in test_images[::-1]:

#     filename = image['filename']

#     # arbitrary prediction, just to store something.
#     q = [1.0, 0.0, 0.0, 0.0]
#     r = [10.0, 0.0, 0.0]

#     submission.append_test(filename, q, r)

for i, data in enumerate(real_test_loader):
    img, label, K = data
    img = img1.cuda()
    t_out, att_out = val_net.forward(img[None])
    print(t_out)
    print(att_out)
    q = [.71, .71, 0.0, 0.0]
    r = [9.0, .1, .1]
    # submission.append_real_test(filename, q, r)

# submission.export(suffix='debug')
print('Submission exported.')
