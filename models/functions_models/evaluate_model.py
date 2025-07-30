import os
import json
import torch
import joblib
import numpy as np
from scipy.special import inv_boxcox

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
     