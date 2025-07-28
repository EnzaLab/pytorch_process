import os 
import json
import joblib
import torch 
from tqdm import tqdm
import pandas as pd 
from scipy.special import inv_boxcox
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
from functions_models.usefull_function_model import *
from functions_models.models_archi import *


def build_data_loader_inference(login, space_buffer, year, month, day, path_to_save, model, pfts, device, batch_size, threshold): 

    if threshold == 0.5: 
        picture = torch.load(os.path.join(path_to_save, f"boxcox_transf_picture_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt"))
    else :    
        picture = torch.load(os.path.join(path_to_save, f"boxcox_transf_picture_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt"))
        
    if model == 1:
        inputs = picture[:, 8:14, :, :].to(device)
    else : 
        inputs = picture[:, :14, :, :].to(device)

    print(inputs.shape)
    dataset = TensorDataset(inputs)

    dataset_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return (dataset_loader)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    login = '/home/elabourdette'

    for image_size in [100]:
        for num_model in [3, 2]:
            space_buffer = int(image_size / 2)
            max_latitude = int(image_size / 4)
            max_longitude = int(image_size / 4)
            date_buffer = 1
            threshold = 0.5
            batch_size = 200
            pfts = True
            training_name = f"training_all_threshold_{threshold}"
            dataset_name = "from_pfts_study_reset15062025"

            model_path = f'{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/{training_name}/model_{num_model}/space_buffer_{space_buffer}_date_buffer_1/'

            # Choix de l'architecture en fonction des flags
            archi_model = None
            if num_model == 1:
                if pfts:
                    if threshold == 0.99:
                        archi_model = "pfts_cnn_lr_7x10e-4/train_pfts_2/MLP_1.pth"
                    elif threshold == 0.5:
                        archi_model = "pfts_[70,50]_lr_7x10e-4_d_0.1/train_pfts_17/MLP_1.pth"
                else:
                    if threshold == 0.99:
                        archi_model = "tchla_[70]_lr_7x10e-4_batch_200/train_1/MLP_1.pth"
                    elif threshold == 0.5:
                        archi_model = "tchla_[70]_lr_7x10e-4_batch_200/train_10/MLP_1.pth"

            elif num_model == 2:
                if pfts:
                    if threshold == 0.99:
                        archi_model = "pfts_cnn_lr_7x10e-4/train_pfts_23/MLP_2.pth"
                    elif threshold == 0.5:
                        archi_model = "pfts_cnn_lr_7x10e-4/train_pfts_20/MLP_2.pth"
                else:
                    if threshold == 0.99:
                        archi_model = "tchla_[70]_lr_7x10e-4_batch_200/train_2/MLP_2.pth"
                    elif threshold == 0.5:
                        archi_model = "tchla_[70]_lr_7x10e-4_batch_200/train_5/MLP_2.pth"
            
            elif num_model == 3:
                if pfts:
                    if threshold == 0.99:
                        print("NO MODEL")
                    elif threshold == 0.5:
                        archi_model = "pfts_cnn_lr_7x10e-4/train_pfts_4/CNN_2.pth"
                else:
                    if threshold == 0.99:
                        print("NO MODEL")
                    elif threshold == 0.5:
                        archi_model = "tchla_[70]_lr_7x10e-4_batch_200/train_6/CNN_2.pth"
                        
            assert archi_model is not None, "Model architecture not defined."

            model_path_full = os.path.join(model_path, archi_model)
            print(model_path, archi_model)

            year = 2002
            month = 5
            for day in tqdm(range(1, 32), desc="Inference"):
                path_input = f"{login}/complex/share/save_training_Enza/inference/{year}/{month}/input/{day}/"

                dataset = build_data_loader_inference(
                    login, space_buffer, year, month, day, path_input,
                    num_model, pfts, device, batch_size, threshold
                )

                # Création du modèle
                if num_model == 1:
                    model_dl = DNN(6, [70], 1, device)
                    if pfts:
                        model_dl = DNN(6, [70, 50], 3, device)
                elif num_model == 2:
                    model_dl = DNN(14, [100], 1, device)
                    if pfts:
                        model_dl = DNN(14, [100, 50], 3, device)
                else:
                    model_dl = ConvRegression(14, 1, 0.3)
                    if pfts:
                        model_dl = ConvRegression(14, 3, 0.3)

                checkpoint = torch.load(model_path_full, map_location=device)
                model_dl.load_state_dict(checkpoint['model_state_dict'])
                model_dl.eval().to(device)

                prediction_test = []
                with torch.no_grad():
                    for images in dataset:
                        images = images[0].to(device)
                        if num_model != 3:
                            images[images == -10] = float('nan')
                            images = images.nanmean(dim=(2, 3))
                        outputs = model_dl(images.float())
                        prediction_test.append(outputs.view(-1).cpu().numpy())

                prediction_set = np.concatenate(prediction_test).reshape(-1, 3 if pfts else 1)

                path_dico = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/{dataset_name}/space_buffer_{space_buffer}_date_buffer_1/threshold_{threshold}/"
                json_path = os.path.join(path_dico, "dico_lambda_MLP.json")
                with open(json_path, "r") as f:
                    dico = json.load(f)

                path_to_save = f"{login}/complex/share/save_training_Enza/inference/{year}/{month}/output/MLP_model{num_model}_after_EGU_{threshold}/{day}/"
                os.makedirs(path_to_save, exist_ok=True)

                if threshold == 0.5:
                    picture_name = f'boxcox_transf_df_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt'
                else:
                    picture_name = f'boxcox_transf_df_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt'

                picture_path = os.path.join(path_input, picture_name)
                picture = torch.load(picture_path)

                if pfts:
                    keys = ["Micro_Chla", "Nano_Chla", "Pico_Chla"]
                    short_keys = ["micro", "nano", "pico"]
                    predictions = {}
                    truths = {}

                    for i, key in enumerate(keys):
                        short_key = short_keys[i]
                        #transfo_path = os.path.join(path_dico, f"{key}_quantile_transformer.pkl")
                        #transformer = joblib.load(transfo_path)

                        pred_i = prediction_set[:, i].reshape(-1, 1)
                        #pred_i = transformer.inverse_transform(pred_i).flatten()

                        mean = dico[f"{key}_standardization_mean"]
                        std = dico[f"{key}_standardization_std"]
                        pred_i = pred_i * std + mean

                        lmbda = dico[f"{key}_boxcox_lmbda"]
                        pred_i = inv_boxcox(pred_i, lmbda) - 1e-4
                        pred_i = np.clip(pred_i, 0, None)

                        predictions[short_key] = pred_i.tolist()

                    pred_tensor = torch.tensor(np.column_stack(list(predictions.values())), dtype=torch.float32).to(device)
                    tensor_map = torch.cat((picture, pred_tensor), dim=1)
                    save_path = os.path.join(path_to_save, f'boxcox_pfts_prediction_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt')
                    print(save_path)
                    torch.save(tensor_map, save_path)

                else:
                    mean = dico[f"TChla_standardization_mean"]
                    std = dico[f"TChla_standardization_std"]
                    pred_i = prediction_set * std + mean
                    
                    #transfo_path = os.path.join(path_dico, "TChla_quantile_transformer.pkl")
                    #transformer = joblib.load(transfo_path)
                    #prediction_nn = transformer.inverse_transform(prediction_set)
                    
                    lmbda = dico[f"TChla_boxcox_lmbda"]
                    pred_i = inv_boxcox(pred_i, lmbda) - 1e-4

                    pred_tensor = torch.tensor(pred_i, dtype=torch.float32).unsqueeze(1).to(device)

                    tensor_map = torch.cat((picture, pred_tensor.squeeze(-1)), dim=1)
                    save_path = os.path.join(path_to_save, f'boxcox_prediction_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt')
                    print(save_path)
                    torch.save(tensor_map, save_path)

if __name__ == "__main__":
    main()    
        

    