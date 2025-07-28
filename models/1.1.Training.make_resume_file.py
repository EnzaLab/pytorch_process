# This code is to run after made more than 30 trains differents. Make a resume file with each metrics per trained model per set. Then, I obtain all score per model. (function calculate_scores from usefull_function_model)

import os
import torch
import pandas as pd
import json
from tqdm import tqdm
import joblib
from functions_models.usefull_function_model import * 
from functions_models.models_archi import *
from functions_models.build_data_loader import build_data_loader
from scipy.special import inv_boxcox
import os
from torch.utils.data import DataLoader, TensorDataset, random_split 

# --------------------- Evaluate Model --------------------------------#
def evaluate_model(model_num, model, dataloader, device, pfts):
    """
    Évalue un modèle en inversant les transformations et calculant les métriques.
    """
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for x, y, _ in dataloader:
            x = x.to(device)
            if model_num != 3:
                x[x == -10] = float('nan')
                x = x.nanmean(dim=(2, 3))
            y_pred = model(x.float())
            y_true_all.append(y)
            y_pred_all.append(y_pred)

    y_true_all = torch.cat(y_true_all, dim=0).detach().cpu()
    y_pred_all = torch.cat(y_pred_all, dim=0).detach().cpu()

    # TO CHANGE if needed
    dataset = "from_pfts_study_reset17072025"
    threshold = 0.99
    space_buffer = 8
    path_dico = f"./dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/"
    print(f"Using path_dico: {path_dico}")

    metrics_list = []

    if pfts:
        keys = ["Micro_Chla", "Nano_Chla", "Pico_Chla"]
        short_keys = ["micro", "nano", "pico"]

        json_path = os.path.join(path_dico, "dico_lambda_MLP.json")
        with open(json_path, "r") as f:
            dico = json.load(f)

        for i, key in enumerate(keys):
            short_key = short_keys[i]

            #transfo_path = os.path.join(path_dico, f"{key}_quantile_transformer.pkl")
            #if not os.path.exists(transfo_path):
            #    raise FileNotFoundError(f"Missing transformer: {transfo_path}")
            #transformer = joblib.load(transfo_path)

            pred_i = y_pred_all[:, i].numpy().reshape(-1, 1)
            true_i = y_true_all[:, i].numpy().reshape(-1, 1)

            #pred_i = transformer.inverse_transform(pred_i).flatten()
            #true_i = transformer.inverse_transform(true_i).flatten()

            mean = dico[f"{key}_standardization_mean"]
            std = dico[f"{key}_standardization_std"]
            pred_i = pred_i * std + mean
            true_i = true_i * std + mean

            lmbda = dico[f"{key}_boxcox_lmbda"]
            pred_i = inv_boxcox(pred_i, lmbda)
            true_i = inv_boxcox(true_i, lmbda)

    
            # if values lower than 10e-5 put it to 0 
            print("Value < 10e-5 : ", pred_i[pred_i < 1e-5].shape)
            pred_i[pred_i < 1e-5] = 0
            true_i[true_i < 1e-5] = 0
            print("Value < 10e-5 after: ", pred_i[pred_i < 1e-5].shape)
            
            metrics = {
                "prediction": short_key,
                "R2": function_r2(pred_i, true_i)[0],
                "MAPD": function_MAPD(pred_i, true_i),
                "RMSD": function_RMSD(pred_i, true_i),
                "SLOPE": function_slope(pred_i, true_i),
            }
            metrics_list.append(metrics)

    else:
        #TChla_path = os.path.join(path_dico, "TChla_quantile_transformer.pkl")
        #quantile_transformer = joblib.load(TChla_path)
        key = "TChla"
        short_key = "tchla"
    
        predictions = {}
        truths = {}
    
        # Charger le dico contenant les stats (mean/std, boxcox lambda)
        json_path = os.path.join(path_dico, "dico_lambda_MLP.json")
        with open(json_path, "r") as f:
            dico = json.load(f)
        prediction_set = y_pred_all.cpu().numpy().reshape(-1, 1)
        true_set = y_true_all.cpu().numpy().reshape(-1, 1)

        # Déstandardisation
        mean = dico[f"{key}_standardization_mean"]
        std = dico[f"{key}_standardization_std"]
        pred_i = prediction_set * std + mean
        true_i = true_set * std + mean

        # Déboxcox
        lmbda = dico[f"{key}_boxcox_lmbda"]
        predictions = inv_boxcox(pred_i, lmbda)
        truths = inv_boxcox(true_i, lmbda)
        
        
        metrics = {
            "prediction": "TChla",
            "R2": function_r2(predictions, truths)[0],
            "MAPD": function_MAPD(predictions, truths),
            "RMSD": function_RMSD(predictions, truths),
            "SLOPE": function_slope(predictions, truths),
        }
        metrics_list.append(metrics)

    return metrics_list

# --------------------- Initialisation --------------------------------#
model_num  = 1
pfts = True
image_size = 16
space_buffer = int(image_size/2)
threshold=0.99
prediction = "pfts_cnn_lr_7x10e-4" if pfts else "tchla_[70]_lr_7x10e-4_batch_200"
batch_size = 200

model_map = {
    1: "MLP_1.pth",
    2: "MLP_2.pth",
    3: "CNN_2.pth"
}
MODEL_FILENAME = model_map.get(model_num)
dataset="from_pfts_study_reset17072025"
BASE_PATH = f"./training_all_threshold_{threshold}_boxcox/model_{model_num}_loss_manual_loss_tchla_cond/space_buffer_{space_buffer}_date_buffer_1/{prediction}"

path_dico = f"./dataset/{dataset}/threshold_{threshold}/space_buffer_{image_size}_date_buffer_1/"

# Collect Data
train_loader, valid_loader, test_loader, df_mini_cubes = build_data_loader(image_size, model_num, pfts, 'cpu', batch_size, dataset, path_dico, threshold)
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
        if pfts: 
            x_value = folder.split("_")[2]
        else :
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
                model_dl = DNN(14, [100, 50], 3, device, p=0.5)
        elif model_num == 3 :
            model_dl = ConvRegression(14, 1, 0.3)
            if pfts :
                model_dl = ConvRegression(14, 3, 0.3)
                
        checkpoint = torch.load(model_path, map_location=device)
        model_dl.load_state_dict(checkpoint["model_state_dict"])
        model_dl.to(device)

        for set_name, dataloader in datasets.items():
            metrics_list = evaluate_model(model_num, model_dl, dataloader, device, pfts)
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

df_test = df_allset[df_allset['set']=='val']
print("Simulation with the best score on the test prediction :", df_test[df_test["Global Score"] == df_test["Global Score"].max()]['nb_training'].item())

txt_path = os.path.join(BASE_PATH, "best_simulation_summary.txt")
with open(txt_path, 'w') as f:
    f.write("Best Simulation Summary (Test Set)\n")
    f.write("-----------------------------------\n")
    f.write(f"nb_training: {df_test[df_test['Global Score'] == df_test['Global Score'].max()]['nb_training'].item()}\n\n")
    f.write("Details:\n")