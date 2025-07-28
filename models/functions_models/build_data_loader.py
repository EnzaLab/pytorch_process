import os
import yaml
import time
import torch 
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import json
from geopy.distance import geodesic
from torch.utils.data import DataLoader, TensorDataset, random_split 
from .collect_mini_cubes import * 
from .apply_transformation import *
from .fill_nan_with_mean import * 
from .add_current import * 
from .apply_bathy_mask import *

def build_data_loader(image_size, model, pfts, device, batch_size, dataset, path_dico, threshold):
    
    space_buffer = int(image_size/2)
    path = f'./dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/'

    if model == 3:
        picture = torch.load(os.path.join(path, f'final_picture_{image_size}x{image_size}_CNN_boxcox.pt'), weights_only=False)
        df = torch.load(os.path.join(path, f'final_df_{image_size}x{image_size}_CNN_boxcox.pt'), weights_only=False)
    else : 
        picture = torch.load(os.path.join(path, f'final_picture_{image_size}x{image_size}_MLP_boxcox.pt'), weights_only=False)
        df = torch.load(os.path.join(path, f'final_df_{image_size}x{image_size}_MLP_boxcox.pt'), weights_only=False)

    valid_mask = ~torch.isnan(df[:, 6])
    picture = picture[valid_mask]
    df = df[valid_mask]

    if model == 1:
        inputs = picture[:, 8:14, :, :].to(device)
    else:
        inputs = picture[:, :14, :, :].to(device)

    target = df[:, 5].to(device)
    if pfts:
        target = df[:, 6:9].to(device)
    set_column = df[:, 9].to(device)
    benchmark = df[:, 11].to(device)
    if pfts:
        benchmark = df[:, 12:15].to(device)

    train_mask = set_column == 0
    val_mask = set_column == 1
    test_mask = set_column == 2

    train_inputs = inputs[train_mask]
    train_target = target[train_mask]
    train_benchmark = benchmark[train_mask]

    # Data augmentation: ensure low_target and high_target each represent 5% of the training data
    data_augmentation = False
    if data_augmentation:
            total_samples = len(train_target)
            print("Total samples before augmentation:", total_samples)
    
            nb_groups = 5
            bin_edges = torch.linspace(train_target.min(), train_target.max(), steps=nb_groups+1).to(train_target.device)
            bin_indices = torch.bucketize(train_target, bin_edges, right=True) - 1 
    
            frequencies = torch.bincount(bin_indices).float()
            frequencies[frequencies == 0] = 1e-6  # éviter la division par 0
    
            duplication_factors = (1.0 / frequencies**0.5)
            duplication_factors *= (total_samples / duplication_factors.sum())  # normalisation globale
            duplication_factors = duplication_factors.int()
    
            augmented_inputs, augmented_targets, augmented_benchmarks = [], [], []
    
            for bin_id in range(10):
                mask = bin_indices == bin_id
                num_duplicates = duplication_factors[bin_id].item()
    
                if num_duplicates > 0 and mask.sum() > 0:
                    augmented_inputs.append(train_inputs[mask].repeat(num_duplicates, 1, 1, 1))
                    augmented_targets.append(train_target[mask].repeat(num_duplicates))
                    augmented_benchmarks.append(train_benchmark[mask].repeat(num_duplicates))
    
            
            train_inputs = torch.cat([train_inputs] + augmented_inputs)
            train_target = torch.cat([train_target] + augmented_targets)
            train_benchmark = torch.cat([train_benchmark] + augmented_benchmarks)
    
            print("Total samples after augmentation:", len(train_target))

    train_dataset = TensorDataset(train_inputs, train_target, train_benchmark)
    valid_dataset = TensorDataset(inputs[val_mask], target[val_mask], benchmark[val_mask])
    test_dataset = TensorDataset(inputs[test_mask], target[test_mask], benchmark[test_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, df



def build_data_loader_select_features(image_size, model, pfts, device, batch_size, dataset, path_dico, threshold, data_to_use):
    space_buffer = int(image_size/2)
    
    if model == 3 : 
        print("TO REFRESH with the new dataset version") 
    else : 
        if  data_to_use == "boxcox": 
            path = f'./dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}'
            picture = torch.load(os.path.join(path, f'final_picture_{image_size}x{image_size}_MLP_boxcox.pt'), weights_only=False)
            df = torch.load(os.path.join(path, f'final_df_{image_size}x{image_size}_MLP_boxcox.pt'), weights_only=False)
        elif data_to_use == "quantile":
            path = f'./dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/quantile'
            picture = torch.load(os.path.join(path, f'final_picture_{image_size}x{image_size}_MLP_quantile.pt'), weights_only=False)
            df = torch.load(os.path.join(path, f'final_df_{image_size}x{image_size}_MLP_quantile.pt'), weights_only=False)

    valid_mask = ~torch.isnan(df[:, 6])
    picture = picture[valid_mask]
    df = df[valid_mask]

    if model == 1:
        inputs = picture[:, 8:14, :, :].to(device)
    else:
        inputs = picture[:, :14, :, :].to(device)

    target = df[:, 5].to(device)
    if pfts:
        target = df[:, 6:9].to(device)
    set_column = df[:, 9].to(device)
    benchmark = df[:, 11].to(device)
    if pfts:
        benchmark = df[:, 12:15].to(device)

    train_mask = set_column == 0
    val_mask = set_column == 1
    test_mask = set_column == 2

    train_inputs = inputs[train_mask]
    train_target = target[train_mask]
    train_benchmark = benchmark[train_mask]

    # Data augmentation: ensure low_target and high_target each represent 5% of the training data
    data_augmentation = False
    if data_augmentation:
            total_samples = len(train_target)
            print("Total samples before augmentation:", total_samples)
    
            nb_groups = 5
            bin_edges = torch.linspace(train_target.min(), train_target.max(), steps=nb_groups+1).to(train_target.device)
            bin_indices = torch.bucketize(train_target, bin_edges, right=True) - 1 
    
            frequencies = torch.bincount(bin_indices).float()
            frequencies[frequencies == 0] = 1e-6  # éviter la division par 0
    
            duplication_factors = (1.0 / frequencies**0.5)
            duplication_factors *= (total_samples / duplication_factors.sum())  # normalisation globale
            duplication_factors = duplication_factors.int()
    
            augmented_inputs, augmented_targets, augmented_benchmarks = [], [], []
    
            for bin_id in range(10):
                mask = bin_indices == bin_id
                num_duplicates = duplication_factors[bin_id].item()
    
                if num_duplicates > 0 and mask.sum() > 0:
                    augmented_inputs.append(train_inputs[mask].repeat(num_duplicates, 1, 1, 1))
                    augmented_targets.append(train_target[mask].repeat(num_duplicates))
                    augmented_benchmarks.append(train_benchmark[mask].repeat(num_duplicates))
    
            
            train_inputs = torch.cat([train_inputs] + augmented_inputs)
            train_target = torch.cat([train_target] + augmented_targets)
            train_benchmark = torch.cat([train_benchmark] + augmented_benchmarks)
    
            print("Total samples after augmentation:", len(train_target))

    train_dataset = TensorDataset(train_inputs, train_target, train_benchmark)
    valid_dataset = TensorDataset(inputs[val_mask], target[val_mask], benchmark[val_mask])
    test_dataset = TensorDataset(inputs[test_mask], target[test_mask], benchmark[test_mask])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader, test_loader, df

