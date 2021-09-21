#Python
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
import models.lossnet as lossnet
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
    if method == 'LLAL':
        models['backbone'].train()
        models['module'].train()
    else:
        models.train()
    global iters

    #for data in tqdm(dataloaders['train'], leave=False, total=len(dataloaders['train'])):
    for data in dataloaders['train']:
        inputs = data[0].cuda()
        labels = data[1].cuda()
        iters += 1


        if method == "LLAL":
            optimizers['backbone'].zero_grad()
            optimizers['module'].zero_grad()
            scores, features = models['backbone'](inputs)
            target_loss = criterion(scores, labels)
            
            if epoch > epoch_loss:
            # After 120 epochs, stop the gradient from the loss prediction module propagated to the target model.
                features[0] = features[0].detach()
                features[1] = features[1].detach()
                features[2] = features[2].detach()
                features[3] = features[3].detach()
           
            pred_loss = models['module'](features)  
            pred_loss = pred_loss.view(pred_loss.size(0))

            m_backbone_loss = torch.sum(target_loss) / target_loss.size(0)
            m_module_loss   = LossPredLoss(pred_loss, target_loss, margin=MARGIN)
            loss            = m_backbone_loss + WEIGHT * m_module_loss
            loss.backward()        
            optimizers['backbone'].step()
            optimizers['module'].step() 
        else:
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
def test(models, dataloaders, mode, method='Entropy'):
    assert mode == 'val' or mode == 'test'
    
    if method == "LLAL":
        models['backbone'].eval()
        models['module'].eval()
    else:
        models.eval()


    total = 0
    correct = 0
    with torch.no_grad():
        for (inputs, labels) in dataloaders[mode]:
            inputs = inputs.cuda()
            labels = labels.cuda()


            if method == "Coreset":
                representation, scores = models(inputs)
            elif method == "LLAL":
                scores, _ = models['backbone'](inputs)
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
        if method == "LLAL":
            schedulers['backbone'].step()
            schedulers['module'].step()
        else:
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
    parser.add_argument('--func', type=str, default='Entropy', help='Select Acquisition Function: Entropy, BALD, VarR. For LLAL, you dont need to use this func')
    parser.add_argument('--dataset', type=str, default='cifar10', help='Select dataset: cifar10, cifar100, Caltech101, Caltech256')
    parser.add_argument('--method', type=str, default='Simple', help='Select method: ENS, DBAL, Simple, Basic, Coreset, LLAL')
    parser.add_argument('--pretrained', type=str, default='False', help='Select using pretrained model of ImageNet')

    args = parser.parse_args()
    return args
##
# Main
if __name__ == '__main__':
    #vis = visdom.Visdom(server='http://localhost', port=9000)
    #plot_data = {'X': [], 'Y': [], 'legend': ['Backbone Loss', 'Module Loss', 'Total Loss']}

    args = get_args()
    print("method: ", args.method)
    print("func: ", args.func)
    print("dataset: ", args.dataset)
    print("pretrained: ", args.pretrained)

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
        
        train_transform = T.Compose(
        [
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])])
        num_classes = 101
        caltech_dataset = Caltech101('../caltech101',  target_type="category" ,download=True, transform=train_transform)
        

        test_dataset, train_dataset = helper.balanced_random_ratio_split(caltech_dataset, 0.3, 0)
        unlabeled_dataset = train_dataset

        print("train_dataset size : ", len(train_dataset))
        print("test_dataset size : ", len(test_dataset))
        
        NUM_TRAIN = 6117 # N
        #SUBSET    = 128 # M
        SUBSET    = 6016 # M
        ADDENDUM  = 1000 # K
        CYCLES = 6
        BATCH = 128    
        
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


    if args.pretrained == 'True':
        pretrained = True
    else:   
        pretrained = False


    print("Batch size: ", BATCH)
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
            models    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()
            
        elif args.method == "ENS":
            models_1    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()
            models_2    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()
            models_3    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()        
        elif args.method == "Coreset":
            print("using coreset netowork")
            models    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()
        elif args.method == "LLAL":
            resnet18    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()
            loss_module = lossnet.LossNet(args.dataset).cuda()
            models      = {'backbone': resnet18, 'module': loss_module}

        else:
            models    = resnet.ResNet18(num_classes, pretrained, args.method, args.dataset).cuda()

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
            
            elif args.method == "LLAL":
                optim_backbone = optim.SGD(models['backbone'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
                optim_module   = optim.SGD(models['module'].parameters(), lr=LR, 
                                    momentum=MOMENTUM, weight_decay=WDECAY)
                sched_backbone = lr_scheduler.MultiStepLR(optim_backbone, milestones=MILESTONES)
                sched_module   = lr_scheduler.MultiStepLR(optim_module, milestones=MILESTONES)

                optimizers = {'backbone': optim_backbone, 'module': optim_module}
                schedulers = {'backbone': sched_backbone, 'module': sched_module} 

                train(models, criterion, optimizers, schedulers, dataloaders, EPOCH, EPOCHL, args.dataset, args.method)
                acc = test(models, dataloaders, mode='test', method=args.method)
                print('Trial {}/{} || Cycle {}/{} || Label set size {}: Test acc {}'.format(trial+1, TRIALS, cycle+1, CYCLES, len(labeled_set), acc))

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
                       
            if cycle != (CYCLES-1): 
                if args.dataset == 'Caltech101':
                    if cycle == 4:
                        subset_num = 1024
                    else:
                        subset_num = SUBSET - 1024 * (cycle + 1)
                    print("subset: ",subset_num)   
                     
                else:
                    subset_num = SUBSET

                random.shuffle(unlabeled_set)
                subset = unlabeled_set[:subset_num]
                print("unlabel set size:", len(unlabeled_set))

                #Create unlabeled dataloader for the unlabeled subset
                unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=BATCH, 
                                          sampler=SubsetSequentialSampler(subset), # more convenient if we maintain the order of subset
                                          pin_memory=True)
            
            
                if args.func == "Random" and args.method == "Simple":
                    labeled_set += list(torch.tensor(subset)[-ADDENDUM:].numpy())
                    unlabeled_set = list(torch.tensor(subset)[:-ADDENDUM].numpy()) + unlabeled_set[subset_num:]
           
                elif args.method == "Coreset":
                    print("Coreset")
                    labeled_loader = dataloaders['train']
                    new_indices = Coreset(models, labeled_loader, unlabeled_loader, num_classes, ADDENDUM)
                    #print("new indices: ", new_indices)
                    select_indices_in_subset = list(torch.tensor(subset)[new_indices].numpy())
                    #print("indices_in_subset:", select_indices_in_subset)
                    labeled_set += select_indices_in_subset

                    non_choosen_set = [idx for idx in subset if idx not in select_indices_in_subset]
                    unlabeled_set = non_choosen_set + unlabeled_set[subset_num:] 
            
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
                    elif args.method == "LLAL":
                        uncertainty = LLAL_uncertainty(models, unlabeled_loader)


                    # Index in ascending order
                    arg = torch.argsort(uncertainty)
                    # Update the labeled dataset and the unlabeled dataset, respectively
                    labeled_set += list(torch.tensor(subset)[arg][-ADDENDUM:].numpy())
                    unlabeled_set = list(torch.tensor(subset)[arg][:-ADDENDUM].numpy()) + unlabeled_set[subset_num:]
                            

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
        elif args.method == 'LLAL':
            torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models['backbone'].state_dict(),
                    'state_dict_module': models['module'].state_dict()
                },
                './' + args.dataset + '/train/weights/active_resnet18_' + args.dataset + '_' + args.func + '_' + args.method + '_trial{}.pth'.format(trial))
        else:
            torch.save({
                    'trial': trial + 1,
                    'state_dict_backbone': models.state_dict(),
                },
                './' + args.dataset + '/train/weights/active_resnet18_' + args.dataset + '_' + args.func + '_' + args.method + '_trial{}.pth'.format(trial))
