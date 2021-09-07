# Python
import os
import random

import time

# Torch
import torch
import numpy as np
from scipy.stats import mode
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler

# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from torchvision.datasets import CIFAR100, CIFAR10
from caltech import Caltech101, Caltech256
# argument
import argparse
from scipy.stats import entropy
# Custom
import models.resnet as resnet

from config import *
from acquistion_function import *
from sampler import SubsetSequentialSampler
import helper
# Seed
random.seed(314)

print(torch.__version__)


##
# Train Utils
iters = 0

#
def train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, method):
    models.train()
    global iters

    #for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1

        optimizers.zero_grad()

        if method == "Coreset":
            representation, scores = models(inputs)
        else:
            scores = models(inputs)
        
        target_loss = criterion(scores, labels)
       

        loss = torch.sum(target_loss) / target_loss.size(0)
        loss.backward()
        optimizers.step()
 


#
def test(models, dataloaders, mode='val', method='Entropy'):
    assert mode == 'val' or mode == 'test'
    models.eval()


    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()


            if method == "Coreset":
                representation, scores = models(inputs)
            else:
                scores = models(inputs)
            
            _, preds = torch.max(scores.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return 100 * correct / total

#
def train(models, criterion, optimizers, schedulers, dataloaders, num_epochs, epoch_loss, datasets, method):
    print('>> Train a Model.')
    best_acc = 0.
    if datasets == 'cifar10':
        p = './cifar10'
    elif datasets == 'cifar100':
        p = './cifar100'
    elif datasets == 'Caltech101':
        p = './Caltech101'
    elif datasets == 'Caltech256':
        p = './Caltech256'
    checkpoint_dir = os.path.join(p, 'train', 'weights')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(num_epochs):
        schedulers.step()


        train_epoch(models, criterion, optimizers, dataloaders, epoch, epoch_loss, method)

        # Save a checkpoint
        if False and epoch % 5 == 4:
            acc = test(models, dataloaders, 'test', method)
            if best_acc < acc:
                best_acc = acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict_backbone': models.state_dict(),
                },
                '%s/active_resnet18_' + p +'.pth' % (checkpoint_dir))
            print('Val Acc: {:.3f} \t Best Acc: {:.3f}'.format(acc, best_acc))
    print('>> Finished.')

#



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--func', type=str, default='Entropy', help='Select Acquisition Function')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Select dataset')
    parser.add_argument('--method', type=str, default='Simple', help='Select method: ENS, DBAL, Simple, Basic, Coreset')

    args = parser.parse_args()
    return args
##
# Main
if __name__ == '__main__':
    #vis = visdom.Visdom(server='http://localhost', port=9000)
    #plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    args = get_args()
    print(args.method)
    print(args.func)
    print(args.dataset)


    # Initailize dataloader
    train_transform = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=4),
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]) # T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) # CIFAR-100
    ])

    if args.dataset == "cifar10":
        train_dataset = CIFAR10('../cifar10', train=True, download=True, transform=train_transform)
        unlabeled_dataset   = CIFAR10('../cifar10', train=True, download=True, transform=test_transform)
        test_dataset  = CIFAR10('../cifar10', train=False, download=True, transform=test_transform)
        num_classes = 10 
    elif args.dataset == "cifar100":
        train_dataset = CIFAR100 ('../cifar100', train=True, download=True, transform=train_transform)
        unlabeled_dataset   = CIFAR100('../cifar100', train=True, download=True, transform=test_transform)
        test_dataset  = CIFAR100('../cifar100', train=False, download=True, transform=test_transform)
        num_classes = 100
    elif args.dataset == 'Caltech101':

        num_classes = 101
        caltech_dataset = Caltech101 ('../caltech101',  target_type="category" ,download=True, transform=train_transform)
        

        test_dataset, train_dataset = helper.balanced_random_ratio_split(caltech_dataset, 0.3, 0)
        unlabeled_dataset = train_dataset

        print("train_dataset size : ", len(train_dataset))
        print("test_dataset size : ", len(test_dataset))
        
        NUM_TRAIN = 6117 # N
        BATCH     = 128 # B
        #SUBSET    = 128 # M
        SUBSET    = 6016 # M
        ADDENDUM  = 1000 # K
        TRIALS = 3
        CYCLES = 6

    elif args.dataset == 'Caltech256':

        num_classes = 256
        caltech_dataset = Caltech256 ('../caltech256',download=True, transform=train_transform)
        

        test_dataset, train_dataset = helper.balanced_random_ratio_split(caltech_dataset, 0.3, 0)
        unlabeled_dataset = train_dataset

        print("train_dataset size : ", len(train_dataset))
        print("test_dataset size : ", len(test_dataset))
        
        NUM_TRAIN = 21531 # N
        BATCH     = 128 # B
        SUBSET    = 128 # M
        #SUBSET    = 10112 # M
        ADDENDUM  = 1000 # K
        TRIALS = 15


    for trial in range(TRIALS):
        # Initialize a labeled dataset by randomly sampling K=ADDENDUM=1,000 data points from the entire dataset.
        indices = list(range(NUM_TRAIN))
        random.shuffle(indices)
        labeled_set = indices[:ADDENDUM]
        unlabeled_set = indices[ADDENDUM:]
        
        train_loader = DataLoader(train_dataset, batch_size=BATCH, 
                                  sampler=SubsetRandomSampler(labeled_set), 
                                  pin_memory=True)
        test_loader  = DataLoader(test_dataset, batch_size=BATCH)
        dataloaders  = {'train': train_loader, 'test': test_loader}
        
        # Model
        if args.method == "DBAL":
            resnet18    = resnet.ResNet18_with_dropout(num_classes).cuda()
            models      = resnet18
        elif args.method == "ENS":
            models_1    = resnet.ResNet18(num_classes).cuda()
            models_2    = resnet.ResNet18(num_classes).cuda()
            models_3    = resnet.ResNet18(num_classes).cuda()        
        elif args.method == "Coreset":
            print("using coreset netowork")
            resnet18    = resnet.ResNet18_coreset(num_classes).cuda()
            models      = resnet18

        else:
            resnet18    = resnet.ResNet18(num_classes).cuda()
            models      = resnet18


        torch.backends.cudnn.benchmark = False

        # Active learning cycles
        for cycle in range(CYCLES):
            start = time.time()
            # Loss, criterion and scheduler (re)initialization
            criterion      = nn.CrossEntropyLoss(reduction='none')
         
            if args.method == "ENS":
                optimizers_1 = optim.SGD(models_1.parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
            
                optimizers_2 = optim.SGD(models_2.parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
           
                optimizers_3 = optim.SGD(models_3.parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)

                schedulers_1 = lr_scheduler.MultiStepLR(optimizers_1, milestones=MILESTONES)
                schedulers_2 = lr_scheduler.MultiStepLR(optimizers_2, milestones=MILESTONES)
                schedulers_3 = lr_scheduler.MultiStepLR(optimizers_3, milestones=MILESTONES)
                
                # Training and test
                train(models_1, criterion, optimizers_1, schedulers_1, dataloaders, EPOCH, EPOCHL, args.dataset, args.method)
                train(models_2, criterion, optimizers_2, schedulers_2, dataloaders, EPOCH, EPOCHL, args.dataset, args.method)
                train(models_3, criterion, optimizers_3, schedulers_3, dataloaders, EPOCH, EPOCHL, args.dataset, args.method)
            
                acc1 = test(models_1, dataloaders, mode='test', method=args.method)
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Model 1 Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc1))
                acc2 = test(models_2, dataloaders, mode='test', method=args.method)
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Model 2 Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc2))
                acc3 = test(models_3, dataloaders, mode='test', method=args.method)
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Model 3 Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc3))

                acc = (acc1 + acc2 + acc3) / 3
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Model avg Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))
            else:
                optimizers = optim.SGD(models.parameters(), lr=LR,
                                     momentum=MOMENTUM, weight_decay=WDECAY)

                schedulers = lr_scheduler.MultiStepLR(optimizers, milestones=MILESTONES)

                # Training and test
                train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, args.dataset, args.method)
                acc = test(models, dataloaders, mode='test', method=args.method)
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))

            ##
            #  Update the labeled dataset via loss prediction-based uncertainty measurement

            # Randomly sample 10000 unlabeled data points
            random.shuffle(unlabeled_set)
            subset = unlabeled_set[:SUBSET]

            #print(subset)           

            # Create unlabeled dataloader for the unlabeled subset
            unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)
            
            
            if args.func == "Random" and args.method == "Simple":
                labeled_set += list(torch.tensor(subset)[-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(subset)[:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
           
            elif args.method == "Coreset":
                print("Coreset")
                labeled_loader = dataloaders['train']
                new_indices = Coreset(models, labeled_loader, unlabeled_loader, num_classes, ADDENDUM)
                #print("new indices: ", new_indices)
                select_indices_in_subset = list(torch.tensor(subset)[new_indices].numpy())
                #print("indices_in_subset:", select_indices_in_subset)
                labeled_set += select_indices_in_subset

                non_choosen_set = [idx for idx in subset if idx not in select_indices_in_subset]
                unlabeled_set = non_choosen_set + unlabeled_set[SUBSET:] 
            
            else:
                # Measure uncertainty of each data points in the subset
                if args.method == "Simple":
                    uncertainty = Simple_uncertainty(models, unlabeled_loader, num_classes)
                elif args.method == "Basic":
                    uncertainty = DBAL_uncertainty(models, unlabeled_loader, dropout_iter, args.func, num_classes)
                elif args.method == "DBAL":
                    uncertainty = DBAL_uncertainty(models, unlabeled_loader, dropout_iter, args.func, num_classes)
                elif args.method == "ENS":
                    uncertainty = ENS_uncertainty(models_1, models_2, models_3, unlabeled_loader, args.func, num_classes)


                # Index in ascending order
                arg = torch.argsort(uncertainty)
                # Update the labeled dataset and the unlabeled dataset, respectively
                labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[SUBSET:]
                            

            # Create a new dataloader for the updated labeled dataset
            dataloaders['train'] = DataLoader(train_dataset, batch_size=BATCH, 
                                              sampler=SubsetRandomSampler(labeled_set), 
                                              pin_memory=True)
                                              
            end = time.time()
            print("execute time：%f s" % (end - start))
        
        # Save a checkpoint
        
        if args.method == 'ENS':
            torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models_1.state_dict(),
                },
                './' + args.dataset + '/train/weights/active_resnet18_' + args.dataset + '_' + args.func + '_' + args.method + 'model1_trial{}.pth'.format(trial))
            torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models_2.state_dict(),
                },
                './' + args.dataset + '/train/weights/active_resnet18_' + args.dataset + '_' + args.func + '_' + args.method + 'model2_trial{}.pth'.format(trial))
            torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models_3.state_dict(),
                },
                './' + args.dataset + '/train/weights/active_resnet18_' + args.dataset + '_' + args.func + '_' + args.method + 'model3_trial{}.pth'.format(trial))

        else:
            torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models.state_dict(),
                },
                './' + args.dataset + '/train/weights/active_resnet18_' + args.dataset + '_' + args.func + '_' + args.method + '_trial{}.pth'.format(trial))
