# This code is the INFERENCE for tchla model 

import pandas as pd 
import xarray as xr
import netCDF4
import os
import numpy as np
import calendar
import torch
import torch.nn as nn
import joblib
from tqdm import tqdm 
import json
from model import * 
from transformation import * 
from Make_clean_df import apply_transfo
import glob
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from scipy.spatial import cKDTree

# --------------------------------------------------------------------------------------------------------------- #
def load_occci_data(login, year, month, date_str, window_size, data_type):
    """
    Charge les données OC-CCI pour la réflectance ou le plancton, puis applique une moyenne glissante.
    """
    
    if data_type not in ['reflectance', 'plankton']:
        raise ValueError("data_type must be either 'reflectance' or 'plankton'")
    
    # path
    if data_type == 'reflectance':
        path = f'{login}/complex/share/c3s/NCDF/OCEANCOLOUR_GLO_BGC_L3_MY_009_107/c3s_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D/{year}/{month}/{date_str}_c3s_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D.nc'
    elif data_type == 'plankton':
        path = f'{login}/complex/share/c3s/NCDF/OCEANCOLOUR_GLO_BGC_L3_MY_009_107/c3s_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D/{year}/{month}/{date_str}_c3s_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D.nc'
    else : 
         raise ValueError("data_type must be either 'reflectance' or 'plankton'")
    print(f'TRY to open :,{path}')
    xr_data = xr.open_mfdataset(path, combine='by_coords', parallel=True)
    xr_data = xr_data.rolling(latitude=window_size, longitude=window_size, center=True).mean()
    df = xr_data.to_dataframe().reset_index()
    df = df.dropna()
    
    return df
    


def make_all_df_same_grid(df_reflectance, df_ssh, df_sst, df_par, df_occci):
    dfs = {
        "reflectance": df_reflectance,
        "ssh": df_ssh,  # Référence
        "sst": df_sst,
        "par": df_par,
        "occci": df_occci
    }

    # Coordonnées de référence (df_ssh)
    lat_ref = df_ssh["latitude"].values
    lon_ref = df_ssh["longitude"].values
    grid_ref = np.array(list(zip(lat_ref, lon_ref)))

    # Nouveau DataFrame fusionné (base : df_ssh)
    merged_df = df_ssh.copy()

    # Seuil de distance en km
    max_distance_km = 16
    max_distance_deg = max_distance_km / 111  # Conversion approximative km → degrés

    # Ajouter les autres variables en récupérant la valeur du point le plus proche
    for name, df in dfs.items():
        if name == "ssh":
            continue  # Déjà la référence

        # Sélectionner les colonnes à ajouter (toutes sauf time, latitude, longitude)
        cols_to_add = [col for col in df.columns if col not in ["time", "latitude", "longitude"]]

        # Construire l'arbre KDTree pour rechercher les plus proches voisins
        points = np.array(list(zip(df["latitude"], df["longitude"])))
        tree = cKDTree(points)

        # Trouver l'indice du point le plus proche et sa distance
        distances, nearest_idx = tree.query(grid_ref, k=1)

        # Ajouter chaque colonne correspondante au DataFrame fusionné
        for col in cols_to_add:
            values = df[col].iloc[nearest_idx].values

            # Remplacer les valeurs par NaN si la distance dépasse le seuil
            values[distances > max_distance_deg] = np.nan
            merged_df[col] = values

    return merged_df
    
def load_cmems(login, year, month, date_str, window_size, data_type):
    """
    Charge les données CMEMS, puis applique une moyenne glissante.
    """
    
    if data_type not in ['ssh', 'sst']:
        raise ValueError("data_type must be either 'ssh' or 'sst'")
    
    # Définition du chemin de base
    if data_type == 'ssh':
        base_path = f'{login}/complex/share/cmems_raw/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D/{year}/{month}/'
        pattern = f'dt_global_allsat_phy_l4_{date_str}*'
    elif data_type == 'sst':
        base_path = f'{login}/complex/share/cmems_raw/SST-GLO-SST-L4-REP-OBSERVATIONS-010-024/{year}/{month}/'
        pattern = f'{date_str}120000*'
    
    # Recherche du fichier correspondant
    file_list = glob.glob(os.path.join(base_path, pattern))
    if not file_list:
        raise FileNotFoundError(f"Aucun fichier trouvé pour le pattern {pattern} dans {base_path}")
    
    # Sélection du premier fichier trouvé
    path = file_list[0]
    print(f"Fichier sélectionné : {path}")
    
    # Chargement des données
    xr_data = xr.open_mfdataset(path, combine='by_coords', parallel=True)
    
    if data_type == 'ssh':
        window_size = 1
        xr_data = xr_data.rolling(latitude=window_size, longitude=window_size, center=True).mean()
        df = xr_data.to_dataframe().reset_index()
        df.dropna(subset=['latitude', 'nv', 'longitude', 'time', 'sla', 'adt','ugos', 'vgos'], inplace=True)
        df = df[['latitude', 'nv', 'longitude', 'time', 'sla', 'adt','ugos', 'vgos']]
        df = df[df['nv']==0]
    elif data_type == 'sst':
        window_size = 3
        xr_data = xr_data.rolling(lat=window_size, lon=window_size, center=True).mean()
        df = xr_data.to_dataframe().reset_index()
        df.dropna(subset=['analysed_sst'], inplace=True)
        df = df[['lat', 'lon', 'time', 'bnds', 'analysed_sst']]
        df = df[df['bnds']==0]
    
    return df

def load_PAR(login, year, month, day, date_str, window_size, data_type):
    
    # path
    if data_type == 'par':
        path = f'{login}/complex/share/c3s/PAR_Acri/ftp.hermes.acri.fr/GLOB/merged/day/{year}/{month}/{day}/L3m_{date_str}__GLOB_4_AV-SWF_PAR_DAY_00.nc'
        print(path)
    else : 
         raise ValueError("data_type must be either 'par'")
    
    xr_data = xr.open_mfdataset(path, combine='by_coords', parallel=True)
    xr_data = xr_data.rolling(lat=window_size, lon=window_size, center=True).mean()
    df = xr_data.to_dataframe().reset_index()
    df.dropna(subset=['PAR_mean'], inplace=True)
    df = df[['lat', 'lon', 'PAR_mean']]
    return df

def apply_bathy_mask(df, interpolator): 
    bathy = []
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        try : 
            bathy_value = interpolator([row['latitude'], row['longitude']])
        except KeyError: 
            bathy_value = interpolator([row['lat'], row['lon']])
        bathy.append(bathy_value)
    df['bathy'] = bathy 
    df = df[df['bathy']<-500]

    return df

class DNN_dropout(nn.Module):
        def __init__(self, input_size, hidden_size_list, output_size, dropout_p, device):
            super(DNN_dropout, self).__init__()
            self.layers = nn.ModuleList()
            self.dropout = nn.Dropout(p=dropout_p)
            self.layers.append(nn.Linear(input_size, hidden_size_list[0]))
            for i in range(len(hidden_size_list) - 1):
                self.layers.append(nn.Linear(hidden_size_list[i], hidden_size_list[i+1]))
            self.output = nn.Linear(hidden_size_list[-1], output_size)
            self.tanh = nn.Tanh()
    
        def forward(self, x):
            for layer in self.layers:
                x = self.dropout(self.tanh(layer(x)))
            x = self.output(x)
            return x

def score_df(df_model_result): 
    results = []
    for sim_num, group in df_model_result.groupby('Statut'):
        min_r2 = group['R2'].min()
        max_r2 = group['R2'].max()
        min_slope = abs(1 - group['Slope']).min()
        max_slope = abs(1 - group['Slope']).max()
        min_mapd = group['MAPD'].min()
        max_mapd = group['MAPD'].max()
        min_rmse = group['RMSD'].min()
        max_rmse = group['RMSD'].max()

        for idx, row in group.iterrows():
            sim_num = row['Numero Simulation']
            r2 = row['R2']
            slope = row['Slope']
            mapd = row['MAPD']
            rmse = row['RMSD']
            status = row['Statut']

            slope_score = (abs(1 - slope) - max_slope) / (min_slope - max_slope)
            r2_score = (r2 - min_r2) / (max_r2 - min_r2)
            mapd_score = (mapd - max_mapd) / (min_mapd - max_mapd)
            rmse_score = (rmse - max_rmse) / (min_rmse - max_rmse)

            global_score = slope_score + r2_score + mapd_score + rmse_score

            results.append({
                'Numero Simulation': sim_num,
                'Statut': status,
                'Slope': row['Slope'],
                'Slope Score': slope_score,
                'R2': row['R2'],
                'R2 Score': r2_score,
                'MAPD': row['MAPD'],
                'MAPD Score': mapd_score,
                'RMSD': row['RMSD'],
                'RMSE Score': rmse_score,
                'Global Score': global_score
            })

    return pd.DataFrame(results)

# --------------------------------------------------------------------------------------------------------------- #    
def make_daily_prediction(login, space_buffer, year, month, day, model_input):
    window_size = int((space_buffer*2)/4)
    month = f'{month:02}' if month < 10 else f'{month}'
    day = f'{day:02}' if day < 10 else f'{day}'
    date_str = f'{year}{month}{day}'
    
    ## Read data ---------------------------------------------------------------------------------------------------------- #

    path_data = f'{login}/complex/share/save_training_Enza/inference/{year}/{month}/{space_buffer}/'
    file_save_input = os.path.join(path_data,f'input/{date_str}_input.csv')
    
    file_save_output = os.path.join(path_data,f'occci_output/{date_str}_occci_prediction.csv')
       
    if os.path.exists(file_save_input):
        input_df = pd.read_csv(file_save_input)
        occci_df = pd.read_csv(file_save_output)
        input_df = input_df.drop(columns=['Unnamed: 0'], errors='ignore')
        occci_df = occci_df.drop(columns=['Unnamed: 0'], errors='ignore')
        occci_df = occci_df[['time','latitude','longitude','CHL','MICRO','NANO','PICO']]
        all_df = pd.merge(input_df, occci_df, on=['latitude', 'longitude', 'time'])
        print('File already download !')
        
    else:
        interpolator = joblib.load(F'{login}/complex/share/save_training_Enza/interpolator_bathy.pkl')

        # OC-CCI input (reflactance) 
        df_reflectance = load_occci_data(login, year, month, date_str, window_size, 'reflectance')
        df_reflectance = apply_bathy_mask(df_reflectance, interpolator)
        # OC-CCI reponse (plankton) 
        df_occci       = load_occci_data(login, year, month, date_str, window_size, 'plankton')
        df_occci = apply_bathy_mask(df_occci, interpolator)
        # SSH 
        df_ssh = load_cmems(login, year, month, date_str, window_size, 'ssh')
        df_ssh = apply_bathy_mask(df_ssh, interpolator)
        # SST
        df_sst = load_cmems(login, year, month, date_str, window_size, 'sst')
        df_sst = apply_bathy_mask(df_sst, interpolator)
        df_sst = df_sst.rename(columns={
        'lon' : 'longitude', 
        'lat' : 'latitude'})
        # PAR
        df_par = load_PAR(login, year, month, day, date_str, window_size, 'par')
        df_par = apply_bathy_mask(df_par, interpolator)
        df_par = df_par.rename(columns={
        'lon' : 'longitude', 
        'lat' : 'latitude'})
    
        df_sst = df_sst[['latitude','longitude','time','analysed_sst']]
        df_ssh = df_ssh[['latitude','longitude','time','sla','adt','ugos','vgos']]
        df_reflectance = df_reflectance[['time', 'latitude', 'longitude', 'RRS412', 'RRS443', 'RRS490', 'RRS510','RRS560', 'RRS665']]
        df_par = df_par[['latitude', 'longitude', 'PAR_mean']]
        df_occci = df_occci [['time', 'latitude', 'longitude', 'CHL', 'MICRO', 'NANO', 'PICO']]
    
        all_df = make_all_df_same_grid(df_reflectance, df_ssh, df_sst, df_par, df_occci) 
    ## Pre treatement inputs data ----------------------------------------------------------------------------------------- #
    all_df = all_df.rename(columns={
            'analysed_sst':'SST',
            'sla':'SLA',
            'adt':'ADT',
            'ugos':'U',
            'vgos':'V',
            'PAR_mean':'PAR'
        })
    all_df, transformed_vars_input = apply_transfo(login, all_df, space_buffer, date_buffer = 1, clean=False, current = False,inference_input = model_input, save = False)
    
    print(transformed_vars_input)
    Data_x = all_df[transformed_vars_input]
    print(Data_x.head())
    ## Select the best model ---------------------------------------------------------------------------------------------- #
    hidden = [70]
    #model_input = "add_['U', 'V', 'ADT', 'SLA', 'SST', 'PAR']"

    model_spe = 'tanh_standardization_MSE_only_maredat_lov'
    Num_simu = 1
    if Num_simu == 'NaN' :
        recap_file_path = f"/home/elabourdette/complex/share/save_training_Enza/training/phyto_one_output_add_physics/model/{model_input}/s_{space_buffer}_d_1/bathy_500/models/deep_learning/hidden_{hidden}_batch_5000_nb_layers_1/without_fine_t/{model_spe}/simulation_recap.csv"
        df_model_result = pd.read_csv(recap_file_path)
        scores_df = score_df(df_model_result)
        max_scores = scores_df.groupby('Statut')['Global Score'].max().reset_index()
        Num_simu = scores_df[(scores_df['Statut'] == 'val') & (scores_df['Global Score'] == max_scores[max_scores['Statut'] == 'val']['Global Score'].values[0])]['Numero Simulation'].item()
    ## Apply the model ---------------------------------------------------------------------------------------------------- #
    input_size = Data_x.shape[1]
    output_size = 1
    batch_size = 5000
    drop_out = 0.1
    
    model_path = f'/home/elabourdette/complex/share/save_training_Enza/training/phyto_one_output_add_physics/model/{model_input}/s_{space_buffer}_d_1/bathy_500/models/deep_learning/hidden_{hidden}_batch_5000_nb_layers_1/without_fine_t/{model_spe}/simulation_{Num_simu}/model.pth'
    print(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dl = DNN_dropout(input_size, hidden, output_size, drop_out, device)
    model_dl.to(device)
    checkpoint = torch.load(model_path)
    model_dl.load_state_dict(checkpoint['model_state_dict'])
    model_dl.eval()
    
    predictions = []
    for index, row in tqdm(Data_x.iterrows(), total=Data_x.shape[0]): 
        input_tensor = torch.tensor(row.values, dtype=torch.float32, device=device)
            
        with torch.no_grad():
            output = model_dl(input_tensor)
            
        predictions.append(output.cpu().item())  # Ramener la prédiction à la CPU et convertir en float
            
    Data_x['predicition'] = predictions
    ## Retreat prediction ---------------------------------------------------------------------------------------------------- #
    with open(f'{login}/complex/share/save_training_Enza/training/phyto_one_output_add_physics/dataset/space_buffer_{space_buffer}_date_buffer_1/dico_lambda.json', 'r') as f:
                    tchla_param = json.load(f)
    mean_tchla = tchla_param['tchla_mean']
    std_tchla = tchla_param['tchla_std']
    prediction_retreated = from_transfo_to_nn(predictions, 1, mean_tchla, std_tchla)
    Data_x['predicition_retreated'] = prediction_retreated

    df_prediction = Data_x.merge(all_df, left_index=True, right_index=True, how='inner')
    df_prediction['CHL - prediction'] = df_prediction['CHL'] - df_prediction['predicition_retreated']

    ## Save the final csv ---------------------------------------------------------------------------------------------------- #
    # Save the inputs
    path_input_data = f'{login}/complex/share/save_training_Enza/inference/{year}/{month}/{space_buffer}/input/'
    if not os.path.exists(path_input_data):
        os.makedirs(path_input_data)
        file_save_input = os.path.join(path_input_data, f'{date_str}_input.csv')
        df_input = df_prediction[['time',	'latitude',	'longitude', 'sla', 'adt', 'ugos', 'vgos', 'RRS412', 'RRS443', 'RRS490', 'RRS510', 'RRS560', 'RRS665','analysed_sst', 'PAR_mean']]# 'analysed_sst_log_std','sla_log_std', 'adt_log_std', 'ugos_log_std', 'vgos_log_std','PAR_mean_log_std'
        df_input.to_csv(file_save_input, index=False)

    path_occci_data = f'{login}/complex/share/save_training_Enza/inference/{year}/{month}/{space_buffer}/occci_output/'
    
    if not os.path.exists(path_occci_data):
        os.makedirs(path_occci_data)
        file_save_occci = os.path.join(path_occci_data, f'{date_str}_occci_prediction.csv')
        df_occci = df_prediction[['time',	'latitude',	'longitude','CHL', 'MICRO', 'NANO', 'PICO']]
        df_occci.to_csv(file_save_occci, index=False)

        # Save the output
    df_output = df_prediction[['time',	'latitude',	'longitude','predicition', 'predicition_retreated', 'CHL', 'MICRO','NANO', 'PICO', 'CHL - prediction']]
    path_output_data = f'{login}/complex/share/save_training_Enza/inference/{year}/{month}/{space_buffer}/prediction/{model_spe}/{model_input}/'
    if not os.path.exists(path_output_data):
                        os.makedirs(path_output_data)
    file_save_input = os.path.join(path_output_data, f'{date_str}_prediction.csv')
    df_output.to_csv(file_save_input)

# --------------------------------------------------------------------------------------------------------------- #
def main():
    ## Parameters 
    login = '/home/elabourdette'
    space_buffer = 16
    window_size = int((space_buffer*2)/4)
    # Which date ?  
    for year in range (2002,2003):
        month = 5 
        #"add_['CURRENT_NORM', 'CURRENT_RAD', 'ADT', 'SLA', 'SST', 'PAR']","add_['CURRENT_NORM', 'CURRENT_RAD']",
        #"add_['U', 'V']",
        for model_input in ["add_['ADT']","add_['SLA']","add_['PAR']","add_['SST']","add_['U', 'V']"]:
            num_days = calendar.monthrange(year, month)[1]
            #range(1,15) num_days + 1
            for day in range(1,num_days + 1):
                make_daily_prediction(login, space_buffer, year, month, day,model_input)

if __name__ == "__main__":
    main()
