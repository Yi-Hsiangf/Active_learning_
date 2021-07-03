# Torch
import torch
import numpy as np
from scipy.stats import mode
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn.functional as F
# Torchvison
import torchvision.transforms as T
import torchvision.models as models
from scipy.stats import entropy

from config import *
# Simple Entropy 
def Simple_uncertainty(models, unlabeled_loader, num_classes):
    models.eval()
    predictions = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            batch_predictions = torch.zeros((BATCH, num_classes)).cuda()
            batch_predictions = models(inputs) # 128 10
            batch_predictions = F.softmax(batch_predictions, dim=1)
            predictions = torch.cat((predictions, batch_predictions + 1e-10), 0) # 10000 10
     
    uncertainty = entropy(predictions.cpu().data.numpy(), base=num_classes, axis=1)
    uncertainty = torch.from_numpy(uncertainty)
    return uncertainty



## DBAL,  Basic
def DBAL_uncertainty(models, unlabeled_loader, dropout_iter, Acquisition_function, num_classes):
    models.eval()
    predictions = torch.tensor([]).cuda()
    full_predictions = torch.tensor([]).cuda()
    class_predictions = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            batch_predictions = torch.zeros((dropout_iter, BATCH, num_classes)).cuda()
       

           
            for i in range(dropout_iter):
                batch_predictions[i,:,:]  = models(inputs) # 100 128 10       1  128 10
                batch_predictions[i,:,:] = F.softmax(batch_predictions[i,:,:], dim=1)
                
            batch_mean_prediction = torch.mean(batch_predictions, 0) #128 10
            predictions = torch.cat((predictions, batch_mean_prediction + 1e-10), 0) # 10000 10             
            if(Acquisition_function == "BALD"): 
                full_predictions = torch.cat((full_predictions, batch_predictions + 1e-10), 1) # 100 128 10 => 100 10000 10       
            elif(Acquisition_function == "VarR"):
                batch_class_predictions = torch.argmax(batch_predictions, dim=-1).float()
                class_predictions = torch.cat((class_predictions, batch_class_predictions + 1e-10), 1) # 100 128 => 100 10000
            
            
            
        if(Acquisition_function == "Entropy"):
            uncertainty = entropy(predictions.cpu().data.numpy(), base=num_classes, axis=1)
            uncertainty = torch.from_numpy(uncertainty)
        elif(Acquisition_function == "BALD"):   
            H = entropy(predictions.cpu().data.numpy(), base=num_classes, axis=1)
            H = torch.from_numpy(H)
            E_H = np.mean(entropy(full_predictions.cpu().data.numpy(), base=num_classes, axis=2), 0)
            E_H = torch.from_numpy(E_H)

            uncertainty = H - E_H
            
        elif(Acquisition_function == "VarR"):
            class_predictions = class_predictions.cpu().numpy()
            Predicted_Class, Mode = mode(class_predictions)
            temp = np.ones(SUBSET) #1 10000  
            temp = temp - Mode / float(dropout_iter)
            num_uncertainty = torch.from_numpy(temp.T)
            uncertainty = torch.squeeze(num_uncertainty)
    
    return uncertainty

def ENS_uncertainty(models_1, models_2, models_3, unlabeled_loader, Acquisition_function, num_classes):
    models_1.eval()
    models_2.eval()
    models_3.eval()
    predictions = torch.tensor([]).cuda()
    full_predictions = torch.tensor([]).cuda()
    class_predictions = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()

            batch_predictions = torch.zeros((3, BATCH, num_classes)).cuda()
       
            batch_predictions[0,:,:]  = F.softmax(models_1(inputs), dim=1) # 1 128 10
            batch_predictions[1,:,:]  = F.softmax(models_2(inputs), dim=1) # 1 128 10
            batch_predictions[2,:,:]  = F.softmax(models_3(inputs), dim=1) # 1 128 10
                
                
            batch_mean_prediction = torch.mean(batch_predictions, 0) #128 10
            predictions = torch.cat((predictions, batch_mean_prediction + 1e-10), 0) # 10000 10             
            if(Acquisition_function == "BALD"): 
                full_predictions = torch.cat((full_predictions, batch_predictions + 1e-10), 1) # 100 128 10 => 100 10000 10       
            elif(Acquisition_function == "VarR"):
                batch_class_predictions = torch.argmax(batch_predictions, dim=-1).float()
                class_predictions = torch.cat((class_predictions, batch_class_predictions + 1e-10), 1) # 100 128 => 100 10000
            
            
            
            
        if(Acquisition_function == "Entropy"):
            uncertainty = entropy(predictions.cpu().data.numpy(), base=num_classes, axis=1)
            uncertainty = torch.from_numpy(uncertainty)
        elif(Acquisition_function == "BALD"):   
            H = entropy(predictions.cpu().data.numpy(), base=num_classes, axis=1)
            H = torch.from_numpy(H)
            E_H = np.mean(entropy(full_predictions.cpu().data.numpy(), base=num_classes, axis=2), 0)
            E_H = torch.from_numpy(E_H)
            
            uncertainty = H - E_H
            
        elif(Acquisition_function == "VarR"):
            class_predictions = class_predictions.cpu().numpy()
            Predicted_Class, Mode = mode(class_predictions)
            temp = np.ones(SUBSET) #1 10000  
            temp = temp - Mode / float(dropout_iter)
            num_uncertainty = torch.from_numpy(temp.T)
            uncertainty = torch.squeeze(num_uncertainty)
    return uncertainty


from scipy.spatial import distance_matrix

def Coreset(models, labeled_loader, unlabeled_loader, num_classes, amount):
    models.eval()
    unlabeled_predictions = torch.tensor([]).cuda()
    labeled_predictions = torch.tensor([]).cuda()
    with torch.no_grad():
        for (inputs, labels) in unlabeled_loader:
            inputs = inputs.cuda()
            batch_predictions = torch.zeros((BATCH, num_classes)).cuda()
            representation, batch_predictions = models(inputs) # 128 10
            unlabeled_predictions = torch.cat((unlabeled_predictions, representation + 1e-10), 0) # 10000 10
            #unlabeled_predictions = torch.cat((unlabeled_predictions, batch_predictions + 1e-10), 0) # 10000 10

        for (inputs, labels) in labeled_loader:
            inputs = inputs.cuda()
            batch_predictions = torch.zeros((BATCH, num_classes)).cuda()
            representation, batch_predictions = models(inputs) # 128 10
            labeled_predictions = torch.cat((labeled_predictions, representation + 1e-10), 0) # 10000 10
            #labeled_predictions = torch.cat((labeled_predictions, batch_predictions + 1e-10), 0) 
        
        new_indices = greedy_k_center(labeled_predictions, unlabeled_predictions, amount) 
        # return a list
    return new_indices

def greedy_k_center(labeled, unlabeled, amount):

        greedy_indices = []

        # get the minimum distances between the labeled and unlabeled examples (iteratively, to avoid memory issues):
        labeled = labeled.cpu().data.numpy()
        unlabeled = unlabeled.cpu().data.numpy()
        min_dist = np.min(distance_matrix(labeled[0, :].reshape((1, labeled.shape[1])), unlabeled), axis=0)
        min_dist = min_dist.reshape((1, min_dist.shape[0]))
        for j in range(1, labeled.shape[0], 100):
            if j + 100 < labeled.shape[0]:
                dist = distance_matrix(labeled[j:j+100, :], unlabeled)
            else:
                dist = distance_matrix(labeled[j:, :], unlabeled)
            min_dist = np.vstack((min_dist, np.min(dist, axis=0).reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))

        # iteratively insert the farthest index and recalculate the minimum distances:
        farthest = np.argmax(min_dist)
        greedy_indices.append(farthest)
        for i in range(amount-1):
            dist = distance_matrix(unlabeled[greedy_indices[-1], :].reshape((1,unlabeled.shape[1])), unlabeled)
            min_dist = np.vstack((min_dist, dist.reshape((1, min_dist.shape[1]))))
            min_dist = np.min(min_dist, axis=0)
            min_dist = min_dist.reshape((1, min_dist.shape[0]))
            farthest = np.argmax(min_dist)
            greedy_indices.append(farthest)

        return greedy_indices 
