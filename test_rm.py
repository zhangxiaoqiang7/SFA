import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import cv2
import pdb

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import rm, combine_dbs
from dataloaders import utils
from networks import deeplab_xception, deeplab_resnet
from dataloaders import custom_transforms as tr
from utils.metrics import runningScore, averageMeter


gpu_id = 1
print('Using GPU: {} '.format(gpu_id))
# Setting parameters
resume_epoch = 25   # Default is 0, change if want to resume
testBatch = 2  # Testing batch size
backbone = 'resnet' # Use xception or resnet as feature extractor,

# classes num is 7
weight = [0.5, 0.5, 1, 4, 1, 2, 2]
#weight = None
#weight = [0.1,0.1,1,4,1,2,2]
n_classes = 7

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]
run_id = 22
save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))

# Network definition
if backbone == 'xception':
    net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=n_classes, os=16, pretrained=True)
elif backbone == 'resnet':
    net = deeplab_resnet.DeepLabv3_plus(nInputChannels=3, n_classes=n_classes, os=16, pretrained=True)
else:
    raise NotImplementedError

modelName = 'deeplabv3plus-' + backbone + '-rm'
criterion = utils.cross_entropy2d

print("Initializing weights from: {}...".format(
    os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
net.load_state_dict(
    torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
               map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

composed_transforms_ts = transforms.Compose([
    tr.FixedResize(size=(480, 640)),#(h,w)
    tr.Normalize(mean=(0.5125, 0.5336, 0.5174), std=(0.2577, 0.2525, 0.2559)),
    tr.ToTensor()])

rm_val = rm.RMSegmentation(split='val', transform=composed_transforms_ts)
testloader = DataLoader(rm_val, batch_size=testBatch, shuffle=False, num_workers=0)

# Setup Metrics
running_metrics_val = runningScore(n_classes=n_classes-1)

num_img_ts = len(testloader)
running_loss_ts = 0.0

# One testing epoch
net.eval()
for ii, sample_batched in enumerate(testloader):
    inputs, labels = sample_batched['image'], sample_batched['label']

    # Forward pass of the mini-batch
    inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
    if gpu_id >= 0:
        inputs, labels = inputs.cuda(), labels.cuda()

    with torch.no_grad():
        outputs = net.forward(inputs)

    predictions = torch.max(outputs, 1)[1]
    
    gt = torch.squeeze(labels).data.cpu().numpy()
    pred = predictions.cpu().numpy()
    
    running_metrics_val.update(gt, pred)
    
    loss = criterion(outputs, labels, ignore_index=n_classes-1, size_average=True, batch_average=True)
    running_loss_ts += loss.item()
    
    # Show images results
    input_imgs = inputs.data.cpu().numpy()
    pred_imgs = utils.decode_seg_map_sequence(pred, dataset='roadmark')
    gt_imgs = utils.decode_seg_map_sequence(gt, dataset='roadmark')
    
    for i in range(input_imgs.shape[0]):
        x = input_imgs[i].transpose(1,2,0)
        #pdb.set_trace()
        normalized = 255.0*(x-x.min())/(x.max()-x.min())
        normalized = cv2.cvtColor(normalized, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(save_dir_root, 'results',str(ii * testBatch+i)+'.jpg'),normalized)
    for i in range(pred_imgs.shape[0]):
        cv2.imwrite(os.path.join(save_dir_root, 'results',str(ii * testBatch+i)+'_p.png'),pred_imgs[i].numpy().transpose(1,2,0))
    for i in range(gt_imgs.shape[0]):
        cv2.imwrite(os.path.join(save_dir_root, 'results',str(ii * testBatch+i)+'_g.png'),gt_imgs[i].numpy().transpose(1,2,0))
    #pdb.set_trace()
    # Print stuff
    if ii % num_img_ts == num_img_ts - 1:
        running_loss_ts = running_loss_ts / num_img_ts
        print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (resume_epoch, ii * testBatch + inputs.data.shape[0]))
        print('Loss: %f' % running_loss_ts)
        running_loss_ts = 0
                    
        # metrics.py
        score, class_iou = running_metrics_val.get_scores()
        for k, v in score.items():
            print('{}: {}'.format(k, v))         
        running_metrics_val.reset()