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
def build_data_loader(login, image_size, model, pfts, device, batch_size, dataset, path_dico, threshold, same_dataset=False, dataset_selected_only_rrs=False):
    #specific_dataset = f"/{dataset}/threshold_{threshold}/"
    space_buffer = int(image_size/2)
    path = f'{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process_split_analysed/dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/'
    #os.makedirs(path, exist_ok=True)

    if same_dataset:
        picture = torch.load(os.path.join(path, f'picture_{image_size}_common.pt'), weights_only=False)
        df = torch.load(os.path.join(path, f'df_{image_size}_common.pt'), weights_only=False)
    elif dataset_selected_only_rrs:
        picture = torch.load(os.path.join(path, f'final_picture_{image_size}x{image_size}_MLP_rrs_selected.pt'), weights_only=False)
        df = torch.load(os.path.join(path, f'final_df_{image_size}x{image_size}_MLP_rrs_selected.pt'), weights_only=False)
    elif model == 3:
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


# --------------------- Evaluate Model --------------------------------#
def evaluate_model(model_num, model, dataloader, device, login, pfts):
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
    path_dico = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/{dataset}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/"
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
login  = "/home/elabourdette"

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
BASE_PATH = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/training_all_threshold_{threshold}_boxcox/model_{model_num}_loss_manual_loss_tchla_cond/space_buffer_{space_buffer}_date_buffer_1/{prediction}"

path_dico = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/{dataset}/threshold_{threshold}/space_buffer_{image_size}_date_buffer_1/"

# Collect Data
train_loader, valid_loader, test_loader, df_mini_cubes = build_data_loader(login, image_size, model_num, pfts, 'cpu', batch_size, dataset, path_dico, threshold)
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
            metrics_list = evaluate_model(model_num, model_dl, dataloader, device, login, pfts)
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
    f.write(best_row.to_string(index=False))