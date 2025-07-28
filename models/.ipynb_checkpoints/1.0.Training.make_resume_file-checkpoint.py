# This code is to run after made more than 30 trains differents. Make a resume file with each metrics per trained model per set. Then, I obtain all score per model. (function calculate_scores from usefull_function_model)

import os
import torch
import pandas as pd
import json
from tqdm import tqdm

from functions_models.usefull_function_model import * 
from functions_models.models_archi import *
from functions_models.build_data_loader import build_data_loader

# --------------------- Evaluate Model --------------------------------#
def evaluate_model(model_num, model, dataloader, device):
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            if model_num != 3 : 
                x[x == -10] = float('nan')
                x = x.nanmean(dim=(2, 3))
            y_pred = model(x.float())
            y_true_all.append(y)
            y_pred_all.append(y_pred)

    y_true_all = torch.cat(y_true_all, dim=0).detach().cpu().numpy()
    y_pred_all = torch.cat(y_pred_all, dim=0).detach().cpu().numpy()

    path_dico = f"{login}/complex/share/save_training_Enza/training/phyto_one_output_add_cnn/dataset/space_buffer_{space_buffer}_date_buffer_1_test/dico_lambda_MLP.json"
    with open(path_dico, 'r') as f:
        dico = json.load(f)
    
    metrics_list = []

    if pfts:
        for i, key in enumerate(["Micro_Chla", "Nano_Chla", "Pico_Chla"]):
            y_pred_i = from_transfo_to_nn(y_pred_all[:, i].flatten().tolist(), 1, dico[f"{key}_mean"], dico[f"{key}_std"])
            y_true_i = from_transfo_to_nn(y_true_all[:, i].flatten().tolist(), 1, dico[f"{key}_mean"], dico[f"{key}_std"])
            metrics = {
                "R2": function_r2(y_pred_i, y_true_i)[0],
                "MAPD": function_MAPD(y_pred_i, y_true_i),
                "RMSD": function_RMSD(y_pred_i, y_true_i),
                "SLOPE": function_slope(y_pred_i, y_true_i),
                "prediction": key
            }
            metrics_list.append(metrics)
    else:
        key = "TChla"
        y_pred = from_transfo_to_nn(y_pred_all.flatten().tolist(), 1, dico[f"{key}_mean"], dico[f"{key}_std"])
        y_true = from_transfo_to_nn(y_true_all.flatten().tolist(), 1, dico[f"{key}_mean"], dico[f"{key}_std"])
        metrics = {
            "R2": function_r2(y_pred, y_true)[0],
            "MAPD": function_MAPD(y_pred, y_true),
            "RMSD": function_RMSD(y_pred, y_true),
            "SLOPE": function_slope(y_pred, y_true),
            "prediction": key
        }
        metrics_list.append(metrics)

    return metrics_list

# --------------------- Initialisation --------------------------------#
login  = "/home/elabourdette"

model_num  = 1
pfts = True
image_size = 32
space_buffer = int(image_size/2)

prediction = "pfts" if pfts else "tchla_[100]_dropout_0.5"

model_map = {
    1: "MLP_1.pth",
    2: "MLP_2.pth",
    3: "CNN_2.pth"
}
MODEL_FILENAME = model_map.get(model_num)

BASE_PATH = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/training/model_{model_num}_stand/space_buffer_{space_buffer}_date_buffer_1/{prediction}_[70, 50]_dropout_0.1"

# Collect Data
train_loader, valid_loader, test_loader, df_mini_cubes = build_data_loader(login, image_size, model_num, pfts, 'cpu', batch_size=2000)
datasets = {
    "train": train_loader,
    "val": valid_loader,
    "test": test_loader
}

results = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Loop per trained model
for folder in os.listdir(BASE_PATH):
    if folder.startswith("train_"):
        x_value = folder.split("_")[1]
        folder_path = os.path.join(BASE_PATH, folder)
        model_path = os.path.join(folder_path, MODEL_FILENAME)

        if model_num == 1: 
            model_dl = DNN(6, [70], 1, device)
            if pfts : 
                model_dl = DNN(6, [70,50], 3, device, p = 0.1)
        elif model_num == 2 : 
            model_dl = DNN(14, [100], 1, device, p=0.5)
            if pfts : 
                model_dl = DNN(14, [120,70], 3, device, p=0.5)
        elif model_num == 3 :
            model_dl = ConvRegression(14, 1, 0.3)
            if pfts :
                model_dl = ConvRegression(14, 3, 0.3)
                
        checkpoint = torch.load(model_path, map_location=device)
        model_dl.load_state_dict(checkpoint["model_state_dict"])
        model_dl.to(device)

        for set_name, dataloader in datasets.items():
            metrics_list = evaluate_model(model_num, model_dl, dataloader, device)
            for metrics in metrics_list:
                metrics.update({
                    "nb_training": x_value,
                    "set": set_name
                })
                results.append(metrics)

# Save
df = pd.DataFrame(results)
df = df.sort_values(by="nb_training", ascending=True)
df_allset = pd.DataFrame()
for set_name in ['train','val','test']:
    df_set = df[df['set'] == set_name]
    df_set = calculate_scores(df_set)
    df_allset = pd.concat([df_allset, df_set], ignore_index=True)
    
df_allset.to_csv(os.path.join(BASE_PATH,"resume.csv"), index=False)

df_test = df_allset[df_allset['set']=='test']
print("Simulation with the best score on the test prediction :", df_test[df_test["Global Score"] == df_test["Global Score"].max()]['nb_training'].item())