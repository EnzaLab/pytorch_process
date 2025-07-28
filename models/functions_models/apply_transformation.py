import os 
import yaml
import json
import torch 
from tqdm import tqdm
import os, json, yaml
import torch.nn.functional as F

#######################################################
###################  MAIN FUNCTION  ###################
#######################################################

def apply_transformation(tensor_mini_cubes, df, login, space_buffer, date_buffer, dataset_name,threshold=0.99, device='cpu', inference=False, save=False):
    import yaml, os, joblib, torch, json
    from sklearn.preprocessing import QuantileTransformer

    tensor_mini_cubes = tensor_mini_cubes.to(device)
    df = df.to(device)

    base_save_path = f"/home/elabourdette/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/from_pfts_study_reset15062025/space_buffer_8_date_buffer_1/threshold_0.99"
    dico_lambda = {}

    path_yaml = f"/home/elabourdette/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/param_no_mod.yaml"
    with open(path_yaml, "r") as file:
        transformations = yaml.safe_load(file)

    input_vars = list(transformations['DATA']['INPUT'].keys())
    output_vars = list(transformations['DATA']['OUTPUT'].keys())
    benchmark_vars = list(transformations['DATA']['BENTCHMARCK'].keys())

    if inference:
        dico_path = os.path.join(base_save_path, "dico_lambda_MLP.json")
        if not os.path.exists(dico_path):
            raise FileNotFoundError(f"Inference mode requires saved stats at {dico_path}")
        with open(dico_path, 'r') as file:
            dico_lambda = json.load(file)

    tensor_vars = ['SLA', 'ADT', 'COSCURRENT', 'SINCURRENT', 'NORMCURRENT', 'SST', 'PAR', 'FSLE'] + \
                  [f'RRS_{w}' for w in [412, 443, 490, 510, 560, 665]] + ['CHL', 'MICRO', 'NANO', 'PICO', 'BATHY']
    var_index = {var: i for i, var in enumerate(tensor_vars)}
    df_index = {'TChla': 5, 'Micro_Chla': 6, 'Nano_Chla': 7, 'Pico_Chla': 8, 'CHL': 11, 'MICRO': 12, 'NANO': 13, 'PICO': 14}

    def process_quantile(var_name, data, is_tensor=True):
        qt_path = os.path.join(base_save_path, f"{var_name}_quantile_transformer.pkl")

        flat_data = data.reshape(-1).clone()
        flat_data[flat_data == 9999.0] = float('nan')
        mask = ~torch.isnan(flat_data)
        flat_data_np = flat_data[mask].cpu().numpy().reshape(-1, 1)

        if inference:
            print("inference")
            qt = joblib.load(dico_lambda[f"{var_name}_quantile_path"])
            transformed = qt.transform(flat_data_np).flatten()
        else:
            qt = QuantileTransformer(output_distribution='normal', random_state=0)
            transformed = qt.fit_transform(flat_data_np).flatten()
            joblib.dump(qt, qt_path)
            dico_lambda[f"{var_name}_quantile_path"] = qt_path

        result_tensor = data.clone().reshape(-1)
        result_tensor[mask] = torch.tensor(transformed, dtype=torch.float32, device=device)
        result_tensor = result_tensor.reshape(data.shape)

        return result_tensor if is_tensor else result_tensor.flatten(), {}

    # -------------------- INPUT VARS --------------------
    for var in input_vars:
        idx = var_index.get(var)
        if idx is None:
            continue

        for transform in transformations['DATA']['INPUT'][var]:
            name = transform['name']

            if name == 'quantile':
                if inference and f"{var}_quantile_path" in dico_lambda:
                    continue
                tensor_mini_cubes[:, idx, :, :], _ = process_quantile(var, tensor_mini_cubes[:, idx, :, :], is_tensor=True)

            elif name in ['normalization', 'standardization']:
                if inference and not any(f"{var}_{k}" in dico_lambda for k in ['mean', 'std', 'min', 'max']):
                    continue
                stats = {k.split("_")[-1]: dico_lambda[k] for k in dico_lambda if k.startswith(var)}
                tensor_mini_cubes[:, idx, :, :], stats = apply_function(name, tensor_mini_cubes[:, idx, :, :], stats if inference else None)
                if save and not inference:
                    for k, v in stats.items():
                        dico_lambda[f"{var}_{k}"] = v

    # -------------------- BENCHMARK VARS --------------------
    for var in benchmark_vars:
        if df.shape[1] <= df_index[var]:
            df = torch.cat((df, torch.full((df.shape[0], 1), torch.nan, device=device)), dim=1)

        idx = var_index.get(var)
        if idx is None:
            continue

        output_data = tensor_mini_cubes[:, idx, :, :].nanmean(dim=(1, 2))

        for transform in transformations['DATA']['OUTPUT'].get(var, []):
            name = transform['name']
            if name == 'quantile' and f"{var}_quantile_path" in dico_lambda:
                output_data, _ = process_quantile(var, output_data, is_tensor=False)
            elif name in ['normalization', 'standardization']:
                stats = {k.split("_")[-1]: dico_lambda[k] for k in dico_lambda if k.startswith(var)}
                output_data, _ = apply_function(name, output_data, stats)

        df[:, df_index[var]] = output_data

    # -------------------- OUTPUT VARS --------------------
    for var in output_vars:
        for transform in transformations['DATA']['OUTPUT'][var]:
            print(var, transform)
            name = transform['name']
            if name == 'quantile' and f"{var}_quantile_path" in dico_lambda:
                df[:, df_index[var]], _ = process_quantile(var, df[:, df_index[var]], is_tensor=False)
            elif name in ['normalization', 'standardization']:
                stats = {k.split("_")[-1]: dico_lambda[k] for k in dico_lambda if k.startswith(var)}
                df[:, df_index[var]], _ = apply_function(name, df[:, df_index[var]], stats)

    if save and not inference:
        dico_path = os.path.join(base_save_path, "dico_lambda_MLP.json")
        with open(dico_path, 'w') as file:
            json.dump(dico_lambda, file)
        print("Dictionary saved at", dico_path)

    return tensor_mini_cubes, df


#######################################################
###################  SUB FUNCTIONS  ###################
#######################################################

###################  APPLY FUNCTION  ###################
from scipy.stats import yeojohnson, boxcox
import torch
import numpy as np

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
            transformed, _ = yeojohnson(data_np, lmbda=stats['lmbda'])
        else:
            transformed, lmbda = yeojohnson(data_np)
            stats['lmbda'] = lmbda
        data_transformed = data.clone()
        data_transformed[~torch.isnan(data)] = torch.tensor(transformed, dtype=data.dtype)
        return data_transformed, stats

    elif name == 'boxcox':
        data_np = data[~torch.isnan(data)].cpu().numpy()
        if np.any(data_np <= 0):
            raise ValueError("Box-Cox transformation requires all values to be strictly positive.")
        if 'lmbda' in stats:
            transformed, _ = boxcox(data_np, lmbda=stats['lmbda'])
        else:
            transformed, lmbda = boxcox(data_np)
            stats['lmbda'] = lmbda
        data_transformed = data.clone()
        data_transformed[~torch.isnan(data)] = torch.tensor(transformed, dtype=data.dtype)
        return data_transformed, stats

    elif name == 'select_over_0.001':
        return data, stats

    else:
        return data, stats

