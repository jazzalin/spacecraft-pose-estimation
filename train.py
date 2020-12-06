# Spacecraft pose estimation - training module
from model.model import MangoNet, TangoNet
from model.dataloader_utils import *
from model.data_transforms import *
from config import Config as CFG

import numpy as np
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from matplotlib import cm

# Random seed
torch.manual_seed(42)

# GPU
if CFG.USE_GPU:
    assert torch.cuda.is_available()
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# LOSS FUNCTIONS

# TRANSLATION
def translation_loss2(t_gt, t_pred, bbox, K):
    # Use 2D detection box to get initial estimate of origin, then regress [du, dv, tz]
    u = (bbox[:, [0]] + bbox[:, [2]]) / 2.0
    v = (bbox[:, [1]] + bbox[:, [3]]) / 2.0
    # print(u, v)
    cx = K[:, [0], [2]]
    cy = K[:, [1], [2]]
    fx = K[:, [0], [0]]
    fy = K[:, [1], [1]]
    # Calculate true deviation from bbox center
    du = fx*t_gt[:, [0]] - u + cx*t_gt[:, [2]]
    dv = fy*t_gt[:, [1]] - v + cy*t_gt[:, [2]]
#     du = fx*t_gt[:, [0]]/t_gt[:, [2]] - u + cx
#     dv = fy*t_gt[:, [1]]/t_gt[:, [2]] - v + cy
    gt = torch.cat((du.float(), dv.float(), t_gt[:, [2]]), -1)
    return torch.mean(torch.norm((gt - t_pred)/torch.norm(gt)))


def translation_loss(t_gt, t_pred):
    return torch.mean(torch.norm((t_gt - t_pred)/torch.norm(t_gt)))


def esa_translation_loss(t_gt, t_pred):
    return torch.sum(torch.norm((t_gt - t_pred)/torch.norm(t_gt)))


# ATTITUDE: Q
def attitude_loss(att_gt, att_pred):
    bs = att_gt.shape[0]
    att_pred = F.normalize(att_pred, p=2, dim=1)
    prod = torch.bmm(att_gt.view(bs, 1, 4), att_pred.view(bs, 4, 1)).reshape(bs, 1)
    loss = 1 - prod ** 2
    return torch.mean(loss)


def esa_attitude_loss(att_gt, att_pred):
    bs = att_gt.shape[0]
    att_pred = F.normalize(att_pred, p=2, dim=1)
    prod = torch.bmm(att_gt.view(bs, 1, 4), att_pred.view(bs, 4, 1)).reshape(bs, 1)
    # loss = 1 - prod ** 2
    loss = 2*torch.acos(torch.abs(prod)).clamp(min=-1.0+1e-7, max=1-1e-7)
    return torch.sum(loss)


# ATTITUDE: PRV
def prv_loss1(q_gt, prv_pred):
    R = ep2dcm(q_gt)
    ss_prv_pred = skew_symmetric(prv_pred)
    R_pred = torch.matrix_exp(ss_prv_pred)
    loss = torch.norm(R_pred - R)
    return loss.mean()


def prv_loss2(q_gt, prv_pred):
    """
    Loss function for Axis-angle attitude regression
        q_gt: ground truth quaternion(s), (bs, 1x4)
        prv_pred: prv output, (bs, 1x3)
    """
    R = ep2dcm(q_gt)
    ss_prv_pred = skew_symmetric(prv_pred)
    R_pred = torch.matrix_exp(ss_prv_pred)
    loss = torch.acos((torch.diagonal(R.transpose(1, 2)@R_pred, dim1=-2, dim2=-1).sum(-1) - 1)/2)
    return loss.mean()


# TRAINING FUNCTIONS
def get_optimizer(net, lr):
    # optimizer = torch.optim.Adam(net.parameters())
    optimizer = torch.optim.SGD(net.parameters(),
                                lr=lr,
                                weight_decay=0.001,
                                momentum=0.9)
    return optimizer


def train_ep(train_loader, net, optimizer_t, optimizer_a, epoch, loss_graph):
    epoch_loss = 0
    epoch_loss_t = 0
    epoch_loss_att = 0
    N = len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, target, K = data
        if CFG.USE_GPU:
            inputs = inputs.cuda()
            target = target.cuda()
            K = K.cuda()
        att_gt = target[:, :4].float()
        t_gt = target[:, 4:7].float()

        optimizer_t.zero_grad()
        optimizer_a.zero_grad()
        t, att = net(inputs)
        # t = net(inputs)
        # att = net(inputs)

#         L_t = translation_loss(t_gt, t)
        L_t = translation_loss2(t_gt, t, target[:, 7:11], K)
        L_att = attitude_loss(att_gt, att)
        total_loss = L_t + CFG.BETA*L_att
        epoch_loss_t += L_t.item()
        epoch_loss_att += L_att.item()

        total_loss.backward()

        optimizer_t.step()
        optimizer_a.step()
        epoch_loss += total_loss.item()

        # Tensorboard viz
        iteration = e * len(train_loader) + i
        loss_graph.add_scalars('training loss', {'total_loss': total_loss.item(),
                                                'translation_loss': L_t.item(),
                                                'attitude_loss': L_att.item()},
                                                iteration)

    return epoch_loss / N, epoch_loss_t / N, epoch_loss_att / N


def train_prv(train_loader, net, optimizer_t, optimizer_a, epoch, loss_graph):
    epoch_loss = 0
    epoch_loss_t = 0
    epoch_loss_att = 0
    N = len(train_loader)
    for i, data in enumerate(train_loader):
        inputs, target, K = data
        if CFG.USE_GPU:
            inputs = inputs.cuda()
            target = target.cuda()
            K = K.cuda()
        att_gt = target[:, :4].float()
        t_gt = target[:, 4:7].float()

        optimizer_t.zero_grad()
        optimizer_a.zero_grad()
        t, att = net(inputs)
#         att = net(inputs)

        L_t = translation_loss2(t_gt, t, target[:, 7:11], K)
        if epoch < 10:
            L_att = prv_loss1(att_gt, att)
        else:
            L_att = prv_loss2(att_gt, att)
            
        total_loss = L_t + CFG.BETA*L_att
        epoch_loss_t += L_t.item()
        epoch_loss_att += L_att.item()

        total_loss.backward()

        optimizer_t.step()
        optimizer_a.step()
        epoch_loss += total_loss.item()

        # Tensorboard viz
        iteration = e * len(train_loader) + i
        loss_graph.add_scalars('training loss', {'total_loss': total_loss.item(),
                                                'translation_loss': L_t.item(),
                                                'attitude_loss': L_att.item()},
                                                iteration)
    return epoch_loss / N, epoch_loss_t / N, epoch_loss_att / N


def evaluate(val_loader, net, loss_graph=None):
    total_loss = 0
    att_loss = 0
    t_loss = 0
    N = len(val_loader)
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            inputs, target, K = data

            if CFG.USE_GPU:
                inputs = inputs.cuda()
                target = target.cuda()
                K = K.cuda()

            att_gt = target[:, :4].float()
            t_gt = target[:, 4:7].float()

            t, att = net(inputs)
            # t = net(inputs)
            # att = net(inputs)

            # Use standard ESA loss functions for evaluation
            # Attitude conversion:
            if CFG.ATT == "prv":
                R_pred = torch.matrix_exp(skew_symmetric(att))
                att = dcm_to_ep(R_pred)

            L_att = esa_attitude_loss(att_gt, att)

            # Translation conversion:
            t_out = origin_reg_conversion(target.detach().cpu().numpy()[0],
                                        K.detach().cpu().numpy()[0],
                                        t.detach().cpu().numpy()[0])
            L_t = esa_translation_loss(t_gt, torch.Tensor(t_out).cuda())

            loss = L_t + L_att
            total_loss += loss.item()
            att_loss += L_att.item()
            t_loss += L_t.item()

            # Tensorboard viz
            # loss_graph.add_scalars('validation loss', {'total_loss': loss.item(),
                                                    #    'translation_loss': L_t.item(),
                                                    #    'attitude_loss': L_att.item()},
                                                    #    i)

    return total_loss/N, t_loss / N, att_loss/N


# VISUALISATION
def visualise_sample(net, data, idx=None):
        N = len(data)
        if idx is None or idx > N:
            idx = np.random.randint(N)
            print("Idx: {}\t sample: {}".format(idx, data.dataset.sample_ids[idx]))

        img1, label1, K1 = data[idx]
        if CFG.USE_GPU:
            img1 = img1.cuda()
        t_out, att_out = net.forward(img1[None])
        
        if CFG.ATT == "prv":
            R_pred = torch.matrix_exp(skew_symmetric(att_out))
            att_out = dcm_to_ep(R_pred)

        t_out = t_out.detach().cpu().numpy()[0]
        att_out = att_out.detach().cpu().numpy()[0]
        img1 = img1.cpu()

        # Plotting groundtruth
        fig = plt.figure()
        image1, label1 = JointRescrop((256, 256))(tF.to_pil_image(img1), label1.numpy())
        fig.add_subplot(1, 2, 1)
        true_origin = data.dataset.visualize(image1, label1, K1, factor=0.6, bbox=True)

        # Plotting predictions
        fig.add_subplot(1, 2, 2)
        # Origin regression from bbox center
        att_out /= np.linalg.norm(att_out)
        t_pred_new = origin_reg_conversion(label1, K1, t_out)
        # origin = dataset.dataset.visualize(image1, np.concatenate((label1[:4], t_pred_new)), K1, factor=0.6, bbox=True)
        origin = data.dataset.visualize(image1, np.concatenate((att_out, t_pred_new)), K1, factor=0.6, bbox=True)
        
        # Prediction errors
        origin_error = np.linalg.norm(np.array(true_origin) - np.array(origin))
        t_error = np.linalg.norm((label1[4:7] - t_pred_new)/np.linalg.norm(label1[4:7]))
        # q_error = 1-np.dot(label1[:4], att_out)**2
        q_error = 2*np.arccos(np.abs(np.dot(label1[:4], att_out)))
        print("Translation error: {} ({})\n gt:\t{}\n pred:\t{}".format(t_error, origin_error, label1[4:7], t_pred_new))
        print("Orientation error: {}\n gt:\t{}\n pred:\t{}".format(q_error, label1[:4], att_out))
        plt.show()



if __name__ == "__main__":

    # Transforms
    train_transform = JointCompose([
        JointRescrop((256,256)),
        JointToTensor(),
        # JointNormalize([0.5], [0.5])
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
    training_dataset = SpeedDataset(
        split="train",
        split_index=CFG.SPLIT_TRAINING_INDEX, # used to make a smaller training set for dev
        speed_root=CFG.DATASET_ROOT,
        annotations_root=CFG.ANNOTATIONS_ROOT,        
        transform=train_transform
    )

    sanity_dataset = SpeedDataset(
        split="train",
        sanity_check=CFG.SANITY_CHECK_INDEX,
        speed_root=CFG.DATASET_ROOT,
        annotations_root=CFG.ANNOTATIONS_ROOT,        
        transform=sanity_transform
    )

    (train_len, test_len) = (int(0.85*len(training_dataset)), int(0.15*len(training_dataset)))
    assert train_len + test_len == len(training_dataset)
    train_dataset, test_dataset = torch.utils.data.random_split(training_dataset, (train_len, test_len))

    # Dataloading
    # NOTE: set shuffle to True!
    train_loader = DataLoader(train_dataset, batch_size=CFG.TRAIN_BS, num_workers=CFG.WORKERS, shuffle=False, drop_last=True)
    sanity_loader = DataLoader(sanity_dataset, batch_size=1, num_workers=CFG.WORKERS, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=CFG.TEST_BS, num_workers=CFG.WORKERS, shuffle=False, drop_last=True)

    # Training
    EPOCH = CFG.EPOCH
    if CFG.ATT == "ep":
        trained_net = MangoNet()
    else:
        trained_net = TangoNet()

    if CFG.USE_GPU:
        trained_net.to(device)

    optimizer_t = get_optimizer(trained_net.t_branch, 0.1)
    optimizer_a = get_optimizer(trained_net, 0.001)
    scheduler_t = torch.optim.lr_scheduler.StepLR(optimizer_t, step_size=25, gamma=0.5)
    scheduler_a = torch.optim.lr_scheduler.StepLR(optimizer_a, step_size=25, gamma=0.5)


    # Switch to training mode
    trained_net.train()

    print("Starting Training...")
    writer = SummaryWriter('runs/mango_run')

    best_loss = float('inf')
    patience = CFG.PATIENCE # number of epochs to wait before early stopping
    trials = 0

    for e in range(EPOCH):
        try:
            if CFG.ATT == "ep":            
                total_loss, loss_t, loss_att = train_ep(train_loader, trained_net, optimizer_t, optimizer_a, e, writer)
            else:
                total_loss, loss_t, loss_att = train_prv(train_loader, trained_net, optimizer_t, optimizer_a, e, writer)
    #         loss = train2(train_loader, trained_net, criterion, optimizer, loss_graph)
    #         loss = train3(train_loader, trained_net, criterion, optimizer_a, loss_graph)
            scheduler_t.step()
            scheduler_a.step()
            print("Epoch: {} Loss: {}, {}, {}".format(e, total_loss, loss_t, loss_att))

            # Save best model
            if total_loss < best_loss:
                trials = 0
                best_loss = total_loss
                torch.save(trained_net.state_dict(), "best.pth")
            else:
                trials += 1
                if trials >= patience:
                    print("Early stopping on epoch: {}".format(e))
                    break
        except KeyboardInterrupt:
            print("Training interrupted...")
            break

    # writer.close()

    # EVALUATION
    # Load best model
    if CFG.ATT == "ep":
        val_net = MangoNet()
    else:
        val_net = TangoNet()

    val_net.load_state_dict(torch.load("best_special.pth"))
    val_net.to(device)
    val_net.eval()
    train_loss = evaluate(train_loader, val_net)
    print("========================= SCORES =========================")
    print("Train error: {}".format(train_loss))
    test_loss = evaluate(test_loader, val_net)
    print("Test error: {}".format(test_loss))
    print("Pose score: {}".format(test_loss[0]))
    print("==========================================================")

    # Visualize results
    show_results = ""
    size = (256, 256)
    vectorScaleFactor = 4
    while show_results == "":
        visualise_sample(val_net, test_dataset)

        show_results = input("Press enter for other sample: ")
        print("----------------------------------------------------------")

