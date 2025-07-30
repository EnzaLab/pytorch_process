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
import joblib
from geopy.distance import geodesic
from torch.utils.data import DataLoader, TensorDataset, random_split 
import numpy as np
from scipy.special import boxcox1p, inv_boxcox

# Personnal function
from functions_models import point_space_boundaries
from functions_models.collect_mini_cubes import *
from functions_models.apply_transformation import apply_transformation
from functions_models.fill_nan_with_mean import fill_nan_with_mean
from functions_models.add_current import add_current
from functions_models.apply_bathy_mask import apply_bathy_mask
from functions_models.usefull_function_model import *
from functions_models.build_data_loader import build_data_loader
from functions_models.train_val_loop import train_val_loop, train_val_loop_l2 
from functions_models.models_archi import *
  
def create_new_training_folder(base_path, pfts=False):
    import re
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    train_dirs = [d for d in existing_dirs if re.match(r'^train_\d+$', d)]
    if pfts : 
        train_dirs = [d for d in existing_dirs if re.match(r'^train_pfts_\d+$', d)]

    indices = [int(re.search(r'\d+', d).group()) for d in train_dirs]
    next_index = max(indices) + 1 if indices else 1

    if pfts :
        new_dir_name = f"train_pfts_{next_index}"
    else : 
        new_dir_name = f"train_{next_index}"
        
    new_dir_path = os.path.join(base_path, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    return new_dir_path


def evaluate_model(dataloader, model_trained, model, device, pfts, path_dico, df_mini_cubes, y_label_name, main_title, path_save_all, data_to_use, suffix, set_type):
    # Map pour filtrage selon set_type
    set_code_map = {'train': 0, 'val': 1, 'test': 2}
    set_code = set_code_map[set_type]

    model_trained.eval()
    prediction_set = []
    true_set = []

    with torch.no_grad():
        for images, labels, _ in dataloader:
            images = images.to(device)
            if model != 3:
                images[images == -10] = float('nan')
                images = images.nanmean(dim=(2, 3))
            outputs = model_trained(images.float())
            prediction_set.append(outputs.cpu())
            true_set.append(labels.cpu())

    prediction_set = torch.cat(prediction_set, dim=0)
    true_set = torch.cat(true_set, dim=0)

    df_set = df_mini_cubes[df_mini_cubes[:, 9] == set_code]
    df_set_lat = df_set[:, 0]

    if pfts:
        keys = ["Micro_Chla", "Nano_Chla", "Pico_Chla"]
        short_keys = ["micro", "nano", "pico"]
    
        predictions = {}
        truths = {}
    
        # Charger le dico contenant les stats (mean/std, boxcox lambda)
        json_path = os.path.join(path_dico, "dico_lambda_MLP.json")
        with open(json_path, "r") as f:
            dico = json.load(f)
    
        for i, key in enumerate(keys):
            short_key = short_keys[i]
            true_i = true_set[:, i].detach().cpu().numpy().reshape(-1, 1)
            pred_i = prediction_set[:, i].detach().cpu().numpy().reshape(-1, 1)
            if data_to_use == "quantile":
                # Charger quantile transformer
                transfo_path = os.path.join(path_dico, f"{key}_quantile_transformer.pkl")
                if not os.path.exists(transfo_path):
                   raise FileNotFoundError(f"Missing transformer: {transfo_path}")
                transformer = joblib.load(transfo_path)
    
                # Inverse quantile transform
                pred_i = transformer.inverse_transform(pred_i).flatten()
                true_i = transformer.inverse_transform(true_i).flatten()

            elif data_to_use == "boxcox": 
                # Déstandardisation
                mean = dico[f"{key}_standardization_mean"]
                std = dico[f"{key}_standardization_std"]
                pred_i = pred_i * std + mean
                true_i = true_i * std + mean
    
                # Déboxcox
                lmbda = dico[f"{key}_boxcox_lmbda"]
                pred_i = inv_boxcox(pred_i, lmbda)
                true_i = inv_boxcox(true_i, lmbda)

                # Thresholrd
                pred_i = pred_i - 10e-4
                true_i = true_i - 10e-4

            pred_i[pred_i < 0.0001] = 0.0001
            true_i[true_i < 0.0001] = 0.0001
            predictions[short_key] = pred_i.tolist()
            truths[short_key] = true_i.tolist()

        # Plot pour chaque PFT
        for pft in short_keys:
            x_label = f'HPLC {pft}-Chla (mg.m$^{{-3}}$)'
            y_label = f'{y_label_name} \n {pft}-Chla (mg.m$^{{-3}}$)'
            output_file = os.path.join(path_save_all, f'{pft}_{suffix}_plot.png')

            plot_score2(
                predictions[pft],
                truths[pft],
                main_title,
                df_set_lat,
                output_link=output_file,
                axis_log=True,
                xlabel=x_label,
                ylabel=y_label,
                PFTS=True
            )
    else:
        key = "TChla"
        short_key = "tchla"
    
        predictions = {}
        truths = {}
    
        json_path = os.path.join(path_dico, "dico_lambda_MLP.json")
        with open(json_path, "r") as f:
            dico = json.load(f)
        prediction_set = prediction_set.cpu().numpy().reshape(-1, 1)
        true_set = true_set.cpu().numpy().reshape(-1, 1)

        if data_to_use == "quantile":
            # Charger quantile transformer
            TChla_path = os.path.join(path_dico, "TChla_quantile_transformer.pkl")
            quantile_transformer = joblib.load(TChla_path)

            predictions = quantile_transformer.inverse_transform(prediction_set)
            truths = quantile_transformer.inverse_transform(true_set)
        
        elif data_to_use == "boxcox": 
            # Déstandardisation
            mean = dico[f"{key}_standardization_mean"]
            std = dico[f"{key}_standardization_std"]
            pred_i = prediction_set * std + mean
            true_i = true_set * std + mean
    
            # Déboxcox
            lmbda = dico[f"{key}_boxcox_lmbda"]
            pred_i = inv_boxcox(pred_i, lmbda)
            true_i = inv_boxcox(true_i, lmbda)
    

        x_label = f'HPLC Total-Chla (mg.m$^{{-3}}$)'
        y_label = f'{y_label_name} \n Total-Chla (mg.m$^{{-3}}$)'
        output_file = os.path.join(path_save_all, f'tchla_{suffix}_plot.png')
        plot_score2(pred_i, true_i , main_title, df_set_lat, output_link=output_file, axis_log=True, xlabel=x_label, ylabel=y_label)
         
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = "from_pfts_study_reset17072025"

    ### Define yours parameters ###
    nb_training = 1
    image_size = 16
    date_buffer = 1
    space_buffer = int(image_size/2)
    print(image_size, space_buffer)
    
    threshold = 0.99
    model = 2
    pfts = False
    data_to_use = 'boxcox'#boxcox or quantile for the moment then add physics simulation
    loss = "MSE_loss"
    #Hyper parameters 
    learning_rate = 7*10e-4
    batch_size = 100
    nb_epoch = 6000
    
    loss_diff_threshold = 0.002
    training_name = f"all_threshold_{threshold}_{data_to_use}"

    # Title figures 
    if model == 3 :
        main_title = f'CNN on {image_size}x{image_size} km with optics and physics input'
        y_label_name = f'CNN with Physics'
        model_save_name = "CNN_2.pth"
    elif model == 2 : 
        main_title = f'MLP on {image_size}x{image_size} km with optics and physics input'
        y_label_name = f'MLP with Physics'
        model_save_name = "MLP_2.pth"
    elif model == 1 : 
        main_title = f'MLP on {image_size}x{image_size} km with optics input'
        y_label_name = f'MLP'
        model_save_name = "MLP_1.pth"

    name_loss_picture = "loss_plot"
    if pfts : 
        name_loss_picture += "_pfts"

    name_mapd_picture = "mapd_plot"
    if pfts : 
        name_mapd_picture += "_pfts"
    
    ############## int((image_size/4))
    max_longitude = int((image_size/4))

    if data_to_use == "boxcox": 
        path_dico = f"./dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}"
    elif data_to_use == "quantile" : 
        path_dico = f"./dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/quantile"
    # Data loader
    train_loader, valid_loader, test_loader, df_mini_cubes = build_data_loader(image_size, model, pfts, 'cpu', batch_size, dataset, path_dico, threshold)

    for training in range(nb_training):
        
        if loss == False : 
            path_save = f"./training_{training_name}/model_{model}/space_buffer_{space_buffer}_date_buffer_{date_buffer}"
        else : 
            path_save = f'./training_{training_name}/model_{model}_{loss}/space_buffer_{space_buffer}_date_buffer_{date_buffer}'
        os.makedirs(path_save, exist_ok=True)
        if pfts : 
            path_save += "/pfts/"
            
        else : 
            path_save += "/tchla/"
                
        path_save_all = create_new_training_folder(path_save, pfts)
        
        # Model Training
        if model == 1: 
            model_dl = DNN(6, [70], 1, device,p=0) 
            if pfts : 
                model_dl = DNN(6, [70,50], 3, device,p=0.1)
        elif model == 2 : 
            model_dl = DNN(14, [100], 1, device, p=0.3)
            if pfts : 
                model_dl = DNN(14, [100, 50], 3, device, p=0.25)
        elif model == 3 :
            model_dl = ConvRegression(14, 1, 0.3)
            if pfts :
                model_dl = ConvRegression(14, 3, 0.3)
                
        model_dl.to(device)
        optimizer = optim.Adam(model_dl.parameters(), lr=learning_rate)

        if loss == "ponderate_loss" : 
            class ExtremeWeightedMSELoss(nn.Module):
                def __init__(self, q_low, q_high, weight_factor=20, pfts=False):
                    super(ExtremeWeightedMSELoss, self).__init__()
                    self.q_low = q_low
                    self.q_high = q_high
                    self.weight_factor = weight_factor
                    self.pfts = pfts
            
                def forward(self, predictions, targets):
                    if not self.pfts:
                        # Une seule dimension
                        extreme_mask = (targets < self.q_low) | (targets > self.q_high)
                    else:
                        # targets: (batch_size, 3)
                        # q_low et q_high: (3,)
                        extreme_mask = (targets < self.q_low) | (targets > self.q_high)  # broadcast automatique
            
                    weights = torch.ones_like(targets)
                    weights[extreme_mask] = self.weight_factor
            
                    errors = (predictions - targets) ** 2
                    weighted_errors = errors * weights
                    return torch.mean(weighted_errors)
            
            ponderate_loss = 0      
            q_low, q_high = compute_target_quantiles(train_loader, quantile=0.05, pfts=pfts)
            criterion = ExtremeWeightedMSELoss(q_low=q_low.to(device), q_high=q_high.to(device), weight_factor=ponderate_loss, pfts=pfts)
        elif loss == "manual_loss": 
            class ManuelMSELoss(nn.Module):
                def __init__(self):
                    super(ManuelMSELoss, self).__init__()
            
                def forward(self, predictions, targets):
                    errors = (predictions - targets) ** 2
                    return torch.mean(errors)

            criterion = ManuelMSELoss() 

        elif loss == "manual_loss_tchla_cond": 
            if pfts == True : 
                # here loss for all communities prediction = Tchla prediction that means weight are equivalent so sum(pico loss, nano loss, micro loss) == tchla loss
                class MSELossTCHLAcondition(nn.Module):  
                    def __init__(self): 
                        super(MSELossTCHLAcondition, self).__init__()
                    
                    def forward(self, predictions, targets): 
                        # Manuel MSE Loss
                        errors = (predictions - targets) ** 2
                        classic_mse_loss = torch.mean(errors) 

                        # Add TCHLA condition 
                        tchla_pred = torch.sum(predictions, dim = 1)
                        tchla_true = torch.sum(targets, dim = 1)
                        diff = tchla_pred - tchla_true
                        mse_tchla = torch.mean(diff**2)

                        total_loss = classic_mse_loss + mse_tchla

                        return total_loss
                
                
                criterion = MSELossTCHLAcondition()
                
            else : 
                print("ERROR - only made when we do pfts.")
        else : 
            print("LOSS MSE")
            criterion = nn.MSELoss() 
        
        LOSS_TRAIN, LOSS_VAL, list_val, model_trained, MAPD_TRAIN, MAPD_VAL = train_val_loop_l2(device, nb_epoch, model_dl, train_loader, valid_loader, optimizer, criterion, loss_diff_threshold, model, pfts)
    
    
        plot_loss(LOSS_TRAIN, LOSS_VAL, main_title, os.path.join(path_save_all, name_loss_picture), "RMSD")
        plot_loss(MAPD_TRAIN, MAPD_VAL, main_title, os.path.join(path_save_all, name_mapd_picture), "MAPD")
    
        evaluate_model(test_loader, model_trained, model, device, pfts, path_dico, df_mini_cubes, y_label_name, main_title, path_save_all, data_to_use, suffix="test", set_type="test")
        evaluate_model(valid_loader, model_trained, model, device, pfts, path_dico, df_mini_cubes, y_label_name, main_title, path_save_all, data_to_use, suffix="val", set_type="val")
        evaluate_model(train_loader, model_trained, model, device, pfts, path_dico, df_mini_cubes, y_label_name, main_title, path_save_all, data_to_use, suffix="train", set_type="train")

        model_save_path = os.path.join(path_save_all, model_save_name)
        torch.save({
        'epoch': nb_epoch, 
        'model_state_dict': model_trained.state_dict(), 
        'optimizer_state_dict': optimizer.state_dict() }, model_save_path)
        print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    main()