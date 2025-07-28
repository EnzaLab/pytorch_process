import os
import json
import yaml
import torch
import joblib
import datetime 
import calendar
import pandas as pd 
import xarray as xr

# Personnal function
from functions_models import apply_bathy_mask
from functions_models import point_space_boundaries
from functions_models.collect_mini_cubes import *
#from functions_models.apply_transformation import apply_transformation
from functions_models.fill_nan_with_mean import fill_nan_with_mean
from functions_models.add_current import add_current
from functions_models.apply_bathy_mask import apply_bathy_mask

############################################################
import os
import json
import yaml
import joblib
import torch
import numpy as np
from scipy.stats import yeojohnson, boxcox
from sklearn.preprocessing import QuantileTransformer

def apply_transformation(tensor_mini_cubes, df, login, space_buffer, date_buffer, threshold=0.99, device='cpu', inference=False, save=False):
    tensor_mini_cubes = tensor_mini_cubes.to(device)
    df = df.to(device)

    # Charger le fichier YAML pour connaître les transformations
    path_yaml = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/param_boxcox.yaml"
    with open(path_yaml, "r") as file:
        transformations = yaml.safe_load(file)

    if inference:
        # Charger les paramètres existants
        dico_path = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/from_pfts_study_reset15062025/space_buffer_{space_buffer}_date_buffer_{date_buffer}/threshold_{threshold}/dico_lambda_MLP.json"
        with open(dico_path, "r") as file:
            dico_lambda = json.load(file)

    input_vars = list(transformations['DATA']['INPUT'].keys())
    output_vars = list(transformations['DATA']['OUTPUT'].keys())
    benchmark_vars = list(transformations['DATA']['BENTCHMARCK'].keys())
    tensor_vars = ['SLA', 'ADT', 'COSCURRENT', 'SINCURRENT', 'NORMCURRENT', 'SST', 'PAR', 'FSLE'] + [f'RRS_{w}' for w in [412, 443, 490, 510, 560, 665]] + ['CHL', 'MICRO', 'NANO', 'PICO','BATHY']
    var_index = {var: i for i, var in enumerate(tensor_vars)}
    df_index = {'CHL': 3, 'MICRO': 4, 'NANO': 5, 'PICO': 6}

    base_save_path = f"{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/from_pfts_study_reset15062025/space_buffer_{space_buffer}_date_buffer_{date_buffer}/threshold_{threshold}"
    os.makedirs(base_save_path, exist_ok=True)

    def process_quantile(var_name, data, is_tensor=True):
        if inference:
            qt_path = dico_lambda.get(f"{var_name}_quantile_path")
            if qt_path:
                qt = joblib.load(qt_path)
                flat_data = data.reshape(-1).clone()
                flat_data[flat_data == 9999.0] = float('nan')
                flat_data = flat_data[~torch.isnan(flat_data)].cpu().numpy().reshape(-1, 1)
                transformed = qt.transform(flat_data).flatten()
                return torch.tensor(transformed, dtype=torch.float32).reshape(data.shape) if is_tensor else torch.tensor(transformed, dtype=torch.float32), {}
        else:
            flat_data = data.reshape(-1).clone()
            flat_data[flat_data == 9999.0] = float('nan')
            flat_data = flat_data[~torch.isnan(flat_data)].cpu().numpy().reshape(-1, 1)
            qt = QuantileTransformer(output_distribution='normal', random_state=0)
            transformed = qt.fit_transform(flat_data).flatten()
            qt_path = os.path.join(base_save_path, f"{var_name}_quantile_transformer.pkl")
            joblib.dump(qt, qt_path)
            dico_lambda[f"{var_name}_quantile_path"] = qt_path
            return torch.tensor(transformed, dtype=torch.float32).reshape(data.shape) if is_tensor else torch.tensor(transformed, dtype=torch.float32), {}

    for var in input_vars:
        idx = var_index.get(var)
        if idx is not None:
            for transform in transformations['DATA']['INPUT'][var]:
                name = transform['name']
                if name == 'quantile':
                    tensor_mini_cubes[:, idx, :, :], _ = process_quantile(var, tensor_mini_cubes[:, idx, :, :], is_tensor=True)
                else:
                    stats = {k.split("_")[-1]: dico_lambda[k] for k in dico_lambda if k.startswith(f"{var}_{name}")}
                    tensor_mini_cubes[:, idx, :, :], _ = apply_function(name, tensor_mini_cubes[:, idx, :, :], stats)


    if save and not inference:
        dico_path = os.path.join(base_save_path, "dico_lambda_MLP.json")
        with open(dico_path, 'w') as file:
            json.dump(dico_lambda, file, indent=2)
        print("Dictionary saved at", dico_path)

    return tensor_mini_cubes, df

def apply_function(name, data, stats=None):
    if stats is None:
        stats = {}

    if name == 'threshold':
        return data + 1e-4, stats
    if name == 'threshold_negative':
        return data - 1e-4, stats
    elif name == 'log10':
        return torch.log10(data), stats
    elif name == 'standardization':
        if stats and 'mean' in stats and 'std' in stats:
            mean, std = stats['mean'], stats['std']
        else:
            mean = data[~torch.isnan(data)].mean()
            std = data[~torch.isnan(data)].std()
            stats = {'mean': mean.item(), 'std': std.item()}
        return (data - mean) / std, stats
    elif name == 'normalization':
        if stats and 'min' in stats and 'max' in stats:
            min_val, max_val = stats['min'], stats['max']
        else:
            min_val = data[~torch.isnan(data)].min()
            max_val = data[~torch.isnan(data)].max()
            stats = {'min': min_val.item(), 'max': max_val.item()}
        return (data - min_val) / (max_val - min_val), stats
    elif name == 'yeojohnson':
        data_np = data[~torch.isnan(data)].cpu().numpy()
        if 'lmbda' in stats:
            transformed = yeojohnson(data_np, lmbda=stats['lmbda'])
        else:
            transformed, lmbda = yeojohnson(data_np)
            stats['lmbda'] = lmbda
        data_transformed = data.clone()
        data_transformed[~torch.isnan(data)] = torch.tensor(transformed, dtype=data.dtype, device=data.device)  # Assurez-vous que le tenseur est sur le même appareil que `data`
        return data_transformed, stats
    elif name == 'boxcox':
        data_np = data[~torch.isnan(data)].cpu().numpy()
        if np.any(data_np <= 0):
            raise ValueError("Box-Cox transformation requires all values to be strictly positive.")
        if 'lmbda' in stats:
            transformed = boxcox(data_np, lmbda=stats['lmbda'])
        else:
            transformed, lmbda = boxcox(data_np)
            stats['lmbda'] = lmbda
        data_transformed = data.clone()
        data_transformed[~torch.isnan(data)] = torch.tensor(transformed, dtype=data.dtype, device=data.device)
        return data_transformed, stats
    elif name == 'select_over_0.001':
        return data, stats
    else:
        return data, stats

############################################################


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    login = '/home/elabourdette'
    
    image_size = 16
    big_space_buffer = 50
    date_buffer = 1
    space_buffer = int(image_size/2)
    max_latitude = int((image_size/4))
    max_longitude = int((image_size/4))
    threshold = 0.99
    year, month = 2002,5
    dataset_name = "from_pfts_study_reset15062025"
    for day in range(1,32): 
        day_str = str(day)
        path = f'{login}/complex/share/save_training_Enza/inference/{year}/{month}/input/{day_str}/'
        picture = torch.load(os.path.join(path, f'tensor_picture_space_{big_space_buffer}_{year}{month}{day}.pt'))
        latitude = torch.load(os.path.join(path, f'tensor_latitude_space_{big_space_buffer}_{year}{month}{day}.pt'))
        longitude = torch.load(os.path.join(path, f'tensor_longitude_space_{big_space_buffer}_{year}{month}{day}.pt'))
        df = torch.load(os.path.join(path, f'tensor_latlon_{year}{month}{day}.pt'))
        picture[picture == 9999.0] = float('nan')
        tensor_mini_cubes, df_mini_cubes  = collect_mini_cubes(df, picture[:,:,:,:], latitude, longitude, space_buffer, max_latitude, max_longitude, device)
        tensor_no_nan, df_no_nan = count_threshold_nan(tensor_mini_cubes, df_mini_cubes, threshold, device)
        tensor_current = add_current(tensor_no_nan)
        tensor_no_cloud = fill_nan_with_mean(tensor_current, device)
        tensor_transf, df_transf = apply_transformation(tensor_no_cloud, df_no_nan, login, space_buffer, date_buffer, threshold, device, inference=True, save=False)
        
        tensor_bathy = apply_bathy_mask(tensor_transf)

        torch.save(tensor_bathy, os.path.join(path, f"boxcox_transf_picture_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt"))
        torch.save(df_transf, os.path.join(path, f"boxcox_transf_df_space_{space_buffer}_{year}{month}{day}_t_{threshold}.pt"))

if __name__ == "__main__":
    main()