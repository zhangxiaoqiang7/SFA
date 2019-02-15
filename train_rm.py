import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
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
use_sbd = False  # Whether to use SBD dataset
nEpochs = 25  # Number of epochs for training
resume_epoch = 0   # Default is 0, change if want to resume

p = OrderedDict()  # Parameters to include in report
p['trainBatch'] = 4  # Training batch size
testBatch = 4  # Testing batch size
useTest = True  # See evolution of the test set when training
nTestInterval = 1 # Run on test set every nTestInterval epochs
snapshot = 5  # Store a model every snapshot epochs
p['nAveGrad'] = 1  # Average the gradient of several iterations
p['lr'] = 0.02 # Learning rate
p['wd'] = 5e-4  # Weight decay
p['momentum'] = 0.9  # Momentum
p['epoch_size'] = 1  # How many epochs to change learning rate
backbone = 'resnet' # Use xception or resnet as feature extractor,
# classes num is 8
weight = [0.5, 0.5, 1, 4, 2, 1, 2, 2] # cls loss weight
#weight = None
#weight = [0.1,0.1,1,4,2,1,2,2]
n_classes = 8
# classes num is 7
weight = [0.5, 0.5, 1, 4, 1, 2, 2]
#weight = None
#weight = [0.1,0.1,1,4,1,2,2]
n_classes = 7
save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    runs = sorted([int(r.split('_')[-1]) for r in runs])
    run_id = runs[-1] if runs else 0
else:
    runs = glob.glob(os.path.join(save_dir_root, 'run', 'run_*'))
    runs = sorted([int(r.split('_')[-1]) for r in runs])
    run_id = runs[-1] + 1 if runs else 0

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


if resume_epoch == 0:
    print("Training deeplabv3+ from scratch...")
else:
    print("Initializing weights from: {}...".format(
        os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth')))
    net.load_state_dict(
        torch.load(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(resume_epoch - 1) + '.pth'),
                   map_location=lambda storage, loc: storage)) # Load all tensors onto the CPU

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

if resume_epoch != nEpochs:
    # Logging into Tensorboard
    log_dir = os.path.join(save_dir, 'models', datetime.now().strftime('%b%d_%H-%M-%S') + '_' + socket.gethostname())
    writer = SummaryWriter(log_dir=log_dir)

    # Use the following optimizer
    optimizer = optim.SGD(net.parameters(), lr=p['lr'], momentum=p['momentum'], weight_decay=p['wd'])
    p['optimizer'] = str(optimizer)

    composed_transforms_tr = transforms.Compose([
        tr.RandomSized((480,640)),#(h,w)
        tr.RandomRotate(15),
        tr.RandomHorizontalFlip(),
        tr.Normalize(mean=(0.5125, 0.5336, 0.5174), std=(0.2577, 0.2525, 0.2559)),
        tr.ToTensor()])

    composed_transforms_ts = transforms.Compose([
        tr.FixedResize(size=(480, 640)),#(h,w)
        tr.Normalize(mean=(0.5125, 0.5336, 0.5174), std=(0.2577, 0.2525, 0.2559)),
        tr.ToTensor()])

    rm_train = rm.RMSegmentation(split='train', transform=composed_transforms_tr)
    rm_val = rm.RMSegmentation(split='val', transform=composed_transforms_ts)

    if use_sbd:
        print("Using SBD dataset")
        sbd_train = sbd.SBDSegmentation(split=['train', 'val'], transform=composed_transforms_tr)
        db_train = combine_dbs.CombineDBs([voc_train, sbd_train], excluded=[voc_val])
    else:
        db_train = rm_train

    trainloader = DataLoader(db_train, batch_size=p['trainBatch'], shuffle=True, num_workers=0)
    testloader = DataLoader(rm_val, batch_size=testBatch, shuffle=False, num_workers=0)
    
    # Setup Metrics
    running_metrics_val = runningScore(n_classes=n_classes-1)

    utils.generate_param_report(os.path.join(save_dir, exp_name + '.txt'), p)

    num_img_tr = len(trainloader)
    num_img_ts = len(testloader)
    running_loss_tr = 0.0
    running_loss_ts = 0.0
    aveGrad = 0
    global_step = 0
    print("Training Network")
    
    # Main Training and Testing Loop
    for epoch in range(resume_epoch, nEpochs):
        start_time = timeit.default_timer()

        if epoch % p['epoch_size'] == p['epoch_size'] - 1:
            lr_ = utils.lr_poly(p['lr'], epoch, nEpochs, 0.9)
            print('(poly lr policy) learning rate: ', lr_)
            optimizer = optim.SGD(net.parameters(), lr=lr_, momentum=p['momentum'], weight_decay=p['wd'])

        net.train()
        for ii, sample_batched in enumerate(trainloader):

            inputs, labels = sample_batched['image'], sample_batched['label']
            # Forward-Backward of the mini-batch
            inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
            global_step += inputs.data.shape[0]

            if gpu_id >= 0:
                inputs, labels = inputs.cuda(), labels.cuda()

            outputs = net.forward(inputs)

            loss = criterion(outputs, labels,ignore_index=n_classes-1, weight=weight, size_average=True, batch_average=True)
            running_loss_tr += loss.item()

            # Print stuff
            if ii % num_img_tr == (num_img_tr - 1):
                running_loss_tr = running_loss_tr / num_img_tr
                writer.add_scalar('data/total_loss_epoch', running_loss_tr, epoch)
                print('[Epoch: %d, numImages: %5d]' % (epoch, ii * p['trainBatch'] + inputs.data.shape[0]))
                print('Loss: %f' % running_loss_tr)
                running_loss_tr = 0
                stop_time = timeit.default_timer()
                print("Execution time: " + str(stop_time - start_time) + "\n")

            # Backward the averaged gradient
            loss /= p['nAveGrad']
            loss.backward()
            aveGrad += 1

            # Update the weights once in p['nAveGrad'] forward passes
            if aveGrad % p['nAveGrad'] == 0:
                writer.add_scalar('data/total_loss_iter', loss.item(), ii + num_img_tr * epoch)
                optimizer.step()
                optimizer.zero_grad()
                aveGrad = 0

            # Show 10 * 3 images results each epoch
            if ii % (num_img_tr // 10) == 0:
                grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy(),dataset='roadmark'),
                     3, normalize=False, range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy(),dataset='roadmark'),
                     3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)

        # Save the model
        if (epoch % snapshot) == snapshot - 1:
            torch.save(net.state_dict(), os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', modelName + '_epoch-' + str(epoch) + '.pth')))

        # One testing epoch
        if useTest and epoch % nTestInterval == (nTestInterval - 1):
            #total_iou = 0.0
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
                
                gt = labels.data.cpu().numpy()
                pred = predictions.cpu().numpy()
                
                running_metrics_val.update(gt, pred)
                
                loss = criterion(outputs, labels, ignore_index=n_classes-1, size_average=True, batch_average=True)
                running_loss_ts += loss.item()

                #total_iou += utils.get_iou(predictions, labels,n_classes=n_classes)

                # Print stuff
                if ii % num_img_ts == num_img_ts - 1:
                    #miou = total_iou / (ii * testBatch + inputs.data.shape[0])
                    running_loss_ts = running_loss_ts / num_img_ts

                    print('Validation:')
                    print('[Epoch: %d, numImages: %5d]' % (epoch, ii * testBatch + inputs.data.shape[0]))
                    writer.add_scalar('data/test_loss_epoch', running_loss_ts, epoch)
                    #writer.add_scalar('data/test_miour', miou, epoch)
                    print('Loss: %f' % running_loss_ts)
                    #print('MIoU: %f\n' % miou)
                    running_loss_ts = 0
                    
                    # metrics.py
                    score, class_iou = running_metrics_val.get_scores()
                    for k, v in score.items():
                        print('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/{}'.format(k), v, epoch)

                    for k, v in class_iou.items():
                        #print('{}: {}'.format(k, v))
                        writer.add_scalar('val_metrics/cls_{}'.format(k), v, epoch)
                        
                    running_metrics_val.reset()


    writer.close()
