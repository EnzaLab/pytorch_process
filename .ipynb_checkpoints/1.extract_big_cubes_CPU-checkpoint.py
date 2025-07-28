import os
import torch
import glob
import numpy as np
import pandas as pd 
from tqdm import tqdm
import torch.nn.functional as F
from geopy.distance import geodesic
from datetime import datetime
import xarray as xr
from models.functions_models.function_extract_cubes import *


def safe_open_nc(path_pattern, description, date, login=None):
    files = glob.glob(path_pattern)
    if not files:
        raise FileNotFoundError(f"Aucun fichier {description} trouvé pour la date {date}")
    file = files[0]
    if login:
        file = os.path.join(login, file)

    ds = xr.open_dataset(file)
    return ds
def point_space_boundaries(lat, lon, buf): 
    """
    MADE for running in a CPU
    """        
    #right
    boundaries = geodesic(kilometers=buf[0]).destination((lat, lon),0)
    lat_r = boundaries.latitude
    #left
    boundaries = geodesic(kilometers=buf[1]).destination((lat, lon),180)
    lat_l = boundaries.latitude
    #up
    boundaries = geodesic(kilometers=buf[0]).destination((lat, lon),90)
    lon_u = boundaries.longitude
    #down
    boundaries = geodesic(kilometers=buf[1]).destination((lat, lon),-90)
    lon_d = boundaries.longitude
    
    tolerance = 0.1
    if lon_d>lon_u: 
        if (lon_u+179.9999<tolerance):
            lon_list = [[lon_d, 179.9999]]
            lat_list = [[lat_l, lat_r]]
        elif (lon_d-179.9999<tolerance): 
            lon_list = [[-179.9999, lon_u]]
            lat_list = [[lat_l, lat_r]]
        else : 
            lon_list = [[lon_d, 179.9999], [-179.9999, lon_u]]
            lat_list = [[lat_l, lat_r], [lat_l, lat_r]]
    else : 
        lat_list = [[lat_l, lat_r]]
        lon_list = [[lon_d, lon_u]]
        

    return lat_list, lon_list

from geopy.distance import geodesic

def get_lon_lat_bounds_geodesic(lat, lon, buffer_km):

    #Newwwwww
    origin = (lat, lon)

    # Latitude Nord / Sud
    lat_r = geodesic(kilometers=buffer_km[0]).destination(origin, bearing=0).latitude
    lat_l = geodesic(kilometers=buffer_km[1]).destination(origin, bearing=180).latitude

    # Longitude Est / Ouest
    lon_u = geodesic(kilometers=buffer_km[0]).destination(origin, bearing=90).longitude
    lon_d = geodesic(kilometers=buffer_km[1]).destination(origin, bearing=270).longitude

    if lon_d > lon_u: 
        lon_list = [[lon_d, 179.9999], [-179.9999, lon_u]]
    else:
        lon_list = [[lon_d, lon_u]]

    # Une seule entrée pour la latitude
    lat_list = [[lat_l, lat_r]]

    return lat_list, lon_list


def selective_data(dataset, var, lat_range, lon_range, lat_name="latitude", lon_name="longitude", method="NaN"):
    if dataset is None:
        # Taille minimale arbitraire (à adapter si besoin)
        dummy_data = np.full((1, 1), 9999.0, dtype=np.float32)
        tensor = torch.tensor(dummy_data)
        lat_values = np.array([lat_range[0]])
        lon_values = np.array([lon_range[0]])
        return tensor, lat_values, lon_values

    # Cas normal avec dataset
    if method == "inverse":
        subset = dataset[var].sel(**{
            lat_name: slice(lat_range[0], lat_range[1]),
            lon_name: slice(lon_range[0], lon_range[1]),
        })
    elif method == "fsle":
        lon_range = [(lon + 360) % 360 if lon < 0 else lon for lon in lon_range]
        subset = dataset[var].sel(**{
            lat_name: slice(lat_range[0], lat_range[1]),
            lon_name: slice(lon_range[0], lon_range[1]),
        })
    else: 
        subset = dataset[var].sel(**{
            lat_name: slice(lat_range[1], lat_range[0]),
            lon_name: slice(lon_range[0], lon_range[1]),
        })

    if subset.size == 0: 
        subset = dataset[var].sel(**{
            lat_name: dataset[lat_name].sel({lat_name: lat_range[0]}, method="nearest"),
            lon_name: dataset[lon_name].sel({lon_name: lon_range[0]}, method="nearest"),
        })

    # Convertir en tensor
    np_data = subset.values.astype(np.float32)
    tensor = torch.from_numpy(np_data)

    lat_values = subset[lat_name].values
    lon_values = subset[lon_name].values

    return tensor, lat_values, lon_values

import numpy as np
import torch

def selective_data(dataset, var, lat_range, lon_range, lat_name="latitude", lon_name="longitude", method="NaN"):
    if dataset is None:
        dummy_data = np.full((1, 1), 9999.0, dtype=np.float32)
        tensor = torch.tensor(dummy_data)
        lat_values = np.array([lat_range[0][0]])
        lon_values = np.array([lon_range[0][0]]) 
        return tensor, lat_values, lon_values
    
    # Latitude
    if method == 'optic':
        lat_range = [[max(lat_range[0]), min(lat_range[0])]]
    else:
        lat_range = [[min(lat_range[0]), max(lat_range[0])]]
        
    lat_slice = slice(lat_range[0][0], lat_range[0][1]) 
    
    # Longitude
    # len(Lon_range) == 2 when near to 180° 
    if len(lon_range) == 2:
        subset_list = []
        lon_values_list = []

        for lon_slice_range in lon_range:
            if method == "fsle":
                lon_slice_range = [(lon + 360) % 360 if lon < 0 else lon for lon in lon_slice_range]
            
            lon_slice = slice(lon_slice_range[0], lon_slice_range[1])
            
            subset_i = dataset[var].sel(**{
                lat_name: lat_slice,
                lon_name: lon_slice,
            })
            if subset_i.size > 0:
                subset_list.append(subset_i)
                lon_values_list.append(subset_i[lon_name].values)
        
        if not subset_list:
            subset = dataset[var].sel(**{
                lat_name: dataset[lat_name].sel({lat_name: lat_range[0]}, method="nearest"),
                lon_name: dataset[lon_name].sel({lon_name: lon_range[0][0]}, method="nearest"),
            })
            lon_values = np.array([subset[lon_name].values])
        else:
            subset = xr.concat(subset_list, dim=lon_name)
            lon_values = np.concatenate(lon_values_list)

    else:
        lon_slice_range = [min(lon_range[0]), max(lon_range[0])]

        if method == "fsle":
            lon_slice_range = [(lon + 360) % 360 if lon < 0 else lon for lon in lon_slice_range]
        
        lon_slice = slice(lon_slice_range[0], lon_slice_range[1])
       
        subset = dataset[var].sel(**{
            lat_name: lat_slice,
            lon_name: lon_slice,
        })
        lon_values = subset[lon_name].values

    # Fallback général si subset vide
    if subset.size == 0:
        subset = dataset[var].sel(**{
            lat_name: dataset[lat_name].sel({lat_name: lat_range[0][0]}, method="nearest"),
            lon_name: dataset[lon_name].sel({lon_name: lon_range[0][0]}, method="nearest"),
        })
        lon_values = subset[lon_name].values

    # Conversion en tensor
    np_data = subset.values.astype(np.float32)
    tensor = torch.from_numpy(np_data)
    lat_values = subset[lat_name].values

    return tensor, lat_values, lon_values


def pad_lat_lon(latitudes, longitudes, max_latitude, max_longitude):
       
    latitudes = torch.tensor(latitudes, dtype=torch.float32)
    longitudes = torch.tensor(longitudes, dtype=torch.float32)

    if latitudes.ndimension() == 1:
        latitudes = latitudes.unsqueeze(0) 
    
    if longitudes.ndimension() == 1:
        longitudes = longitudes.unsqueeze(0)  

    if latitudes.ndimension() == 0:
        latitudes = latitudes.unsqueeze(0).unsqueeze(0)
    
    if longitudes.ndimension() == 0:
        longitudes = longitudes.unsqueeze(0).unsqueeze(0)
    
    if latitudes.size(1) < max_latitude:
        padding_latitude = max_latitude - latitudes.size(1)
        padding_tensor_latitude = torch.full((latitudes.size(0), padding_latitude), 9999, dtype=torch.float32)
        latitudes = torch.cat([latitudes, padding_tensor_latitude], dim=1)

    if longitudes.size(1) < max_longitude:
        padding_longitude = max_longitude - longitudes.size(1)
        padding_tensor_longitude = torch.full((longitudes.size(0), padding_longitude), 9999, dtype=torch.float32)
        longitudes = torch.cat([longitudes, padding_tensor_longitude], dim=1)

    return latitudes, longitudes


def pad_to_max_size(data, max_latitude, max_longitude):
    if data.ndimension() < 3:
        if data.ndimension() == 1:
                data = data.unsqueeze(0).unsqueeze(0)
        elif data.ndimension() == 2:
                data = data.unsqueeze(0)
    
    if data.shape[1] < max_latitude:
                padding_latitude = max_latitude - data.shape[1]
                data = F.pad(data, (0, 0, 0, padding_latitude), value=9999)
            
    if data.shape[2] < max_longitude:
                padding_longitude = max_longitude - data.shape[2]
                data = F.pad(data, (0, padding_longitude, 0, 0), value=9999)
        
    return data

def apply_physical_data(login, df, buf, max_latitude, max_longitude, max_time): 
    path_plk = f'{login}/complex/share/c3s/NCDF/OCEANCOLOUR_GLO_BGC_L3_MY_009_107/c3s_obs-oc_glo_bgc-plankton_my_l3-multi-4km_P1D'
    path_rrs = f'{login}/complex/share/c3s/NCDF/OCEANCOLOUR_GLO_BGC_L3_MY_009_107/c3s_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D'
    path_sst = f'{login}/complex/share/cmems_raw/SST-GLO-SST-L4-REP-OBSERVATIONS-010-024/'
    path_sla = f'{login}/complex/share/cmems_raw/SEALEVEL_GLO_PHY_L4_MY_008_047/cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D/'
    path_par = f'{login}/complex/share/c3s/PAR_Acri/ftp.hermes.acri.fr/GLOB/merged/day/'
    path_fsle = f'{login}/complex/share/c3s/Aviso_FSLE/ftp-access.aviso.altimetry.fr/value-added/lyapunov/delayed-time/global'
    path_bathy_completed = f'{login}/complex/share/save_training_Enza/GEBCO_2023_sub_ice_topo.nc'
    
    tensor_list = [] 
    latitudes_list = []
    longitudes_list = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Traitement des données"):
        lat = row['Lat']
        lon = row['Lon']
        #lat_inter, lon_inter = point_space_boundaries(lat, lon, buf)
        lat_inter, lon_inter = get_lon_lat_bounds_geodesic(lat, lon, buf)
        date = row['date']
        year, month, day = from_date_to_year_month_day(date)

        try:
            path_rrs_completed = os.path.join(path_rrs, f"{year}/{month}/{year}{month}{day}_c3s_obs*.nc")
            path_plk_completed = os.path.join(path_plk, f"{year}/{month}/{year}{month}{day}_c3s_obs*.nc")
            ds_rrs = safe_open_nc(path_rrs_completed, "RRS", date)
            ds_plk = safe_open_nc(path_plk_completed, "PLK", date)
        except Exception:
            ds_rrs = None
            ds_plk = None
        try:
            path_sla_completed = os.path.join(path_sla, f"{year}/{month}/dt_global_allsat_phy_l4_{year}{month}{day}*.nc")
            ds_sla = safe_open_nc(path_sla_completed, "SLA", date)
        except Exception:
            ds_sla = None

        try:
            path_sst_completed = os.path.join(path_sst, f"{year}/{month}/{year}{month}{day}*.nc")
            ds_sst = safe_open_nc(path_sst_completed, "SST", date)
        except Exception:
            ds_sst = None

        try:
            path_par_completed = os.path.join(path_par, f"{year}/{month}/{day}/L3m_{year}{month}{day}*.nc")
            ds_par = safe_open_nc(path_par_completed, "PAR", date)
        except Exception:
            ds_par = None

        try:
            ds_bathy = xr.open_dataset(path_bathy_completed)
        except Exception:
            ds_bathy = None

        try:
            path_fsle_completed = os.path.join(path_fsle, f"{year}/dt_global_allsat_madt_fsle_{year}{month}{day}*.nc")
            ds_fsle = safe_open_nc(path_fsle_completed, "FSLE", date, login=login)
        except Exception:
            ds_fsle = None

        RRS412_subset, RRS_lat, RRS_lon = selective_data(ds_rrs, "RRS412", lat_inter, lon_inter, method="optic")
        RRS443_subset, _, _ = selective_data(ds_rrs, "RRS443", lat_inter, lon_inter, method="optic")
        RRS490_subset, _, _ = selective_data(ds_rrs, "RRS490", lat_inter, lon_inter, method="optic")
        RRS510_subset, _, _ = selective_data(ds_rrs, "RRS510", lat_inter, lon_inter, method="optic")
        RRS560_subset, _, _ = selective_data(ds_rrs, "RRS560", lat_inter, lon_inter, method="optic")
        RRS665_subset, _, _ = selective_data(ds_rrs, "RRS665", lat_inter, lon_inter, method="optic")

        CHL_subset, CHL_lat, CHL_lon = selective_data(ds_plk, "CHL", lat_inter, lon_inter, method="optic")
        MICRO_subset, _, _ = selective_data(ds_plk, "MICRO", lat_inter, lon_inter, method="optic")
        NANO_subset, _, _ = selective_data(ds_plk, "NANO", lat_inter, lon_inter, method="optic")
        PICO_subset, _, _ = selective_data(ds_plk, "PICO", lat_inter, lon_inter, method="optic")

        sla_subset, SLA_lat, SLA_lon = selective_data(ds_sla, "sla", lat_inter, lon_inter)
        adt_subset, _, _ = selective_data(ds_sla, "adt", lat_inter, lon_inter)
        ugos_subset, _, _ = selective_data(ds_sla, "ugos", lat_inter, lon_inter)
        vgos_subset, _, _ = selective_data(ds_sla, "vgos", lat_inter, lon_inter)

        sst_subset, SST_lat, SST_lon = selective_data(ds_sst, "analysed_sst", lat_inter, lon_inter, "lat", "lon")
        par_subset, PAR_lat, PAR_lon = selective_data(ds_par, "PAR_mean", lat_inter, lon_inter, "lat", "lon", method="optic")
        fsle_subset, FSLE_lat, FSLE_lon = selective_data(ds_fsle, "fsle_max", lat_inter, lon_inter, "lat", "lon", method="fsle") 

        bathy_subset, BATHY_lat, BATHY_lon = selective_data(ds_bathy, 'elevation', lat_inter, lon_inter, "lat", "lon")
        bathy_subset = bathy_subset[::10,::10]
        BATHY_lat = BATHY_lat[::10]
        BATHY_lon = BATHY_lon[::10]


        # Pad des coordonnées
        RRS_lat, RRS_lon = pad_lat_lon(RRS_lat, RRS_lon, max_latitude, max_longitude)
        CHL_lat, CHL_lon = pad_lat_lon(CHL_lat, CHL_lon, max_latitude, max_longitude)
        SLA_lat, SLA_lon = pad_lat_lon(SLA_lat, SLA_lon, max_latitude, max_longitude)
        SST_lat, SST_lon = pad_lat_lon(SST_lat, SST_lon, max_latitude, max_longitude)
        PAR_lat, PAR_lon = pad_lat_lon(PAR_lat, PAR_lon, max_latitude, max_longitude)
        FSLE_lat, FSLE_lon = pad_lat_lon(FSLE_lat, FSLE_lon, max_latitude, max_longitude)
        BATHY_lat, BATHY_lon = pad_lat_lon(BATHY_lat, BATHY_lon, max_latitude, max_longitude)

        latitudes_list.extend([SLA_lat, SST_lat, PAR_lat, FSLE_lat, RRS_lat, CHL_lat, BATHY_lat])
        longitudes_list.extend([SLA_lon, SST_lon, PAR_lon, FSLE_lon, RRS_lon, CHL_lon, BATHY_lon])

        # Padding des données
        padded_data = []
        for var in [
            sla_subset, adt_subset, ugos_subset, vgos_subset,
            sst_subset, par_subset, fsle_subset,
            RRS412_subset, RRS443_subset, RRS490_subset, RRS510_subset,
            RRS560_subset, RRS665_subset,
            CHL_subset, MICRO_subset, NANO_subset, PICO_subset,
            bathy_subset
        ]:
            if var is None:
                padded = torch.full((max_latitude, max_longitude), 9999.0)
            else:
                padded = pad_to_max_size(var, max_latitude, max_longitude)
            padded_data.append(padded)
        
        for i, t in enumerate(padded_data):
            if not isinstance(t, torch.Tensor):
                raise TypeError(f"Élément {i} de padded_data n’est pas un Tensor mais {type(t)}")

        data_tensor = torch.stack(padded_data, dim=0)
        tensor_list.append(data_tensor)

    df_filtered = df.reset_index(drop=True)
    final_tensor = torch.cat(tensor_list, dim=0)

    latitudes_tensor = torch.stack(latitudes_list, dim=0).permute(1, 0, 2)
    longitudes_tensor = torch.stack(longitudes_list, dim=0).permute(1, 0, 2)
        
    final_tensor_test = final_tensor.reshape(df_filtered.shape[0], 18, max_time, max_latitude, max_longitude)
    latitudes_tensor_test = latitudes_tensor.reshape(df_filtered.shape[0], 7, max_latitude)
    longitudes_tensor_test = longitudes_tensor.reshape(df_filtered.shape[0], 7, max_longitude)

    return final_tensor, df_filtered, latitudes_tensor, longitudes_tensor

   

def main(): 
    login = '/home/elabourdette'
    space_buffer = 200
    date_buffer  = 1

    df = pd.read_csv(f'{login}/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/pfts_study_data/make_global_df/finale_merge_data_17072025.csv')
    df = df.rename(columns={ "TChla":'tchla', "lat":'Lat', "lon":'Lon', "Date":"date"})
    df['date'] = pd.to_datetime(df[['Year', 'Month', 'Day']])
    df["date"] = df["date"].astype(str)
    df = df[df['tchla']>0.001]
    df = df[df['tchla']<15]
    buf = [space_buffer, space_buffer]
    df = df[(df['Lat'] != 'NAN') & (df['Lon'] != 'NAN')]
    max_latitude = 100
    max_longitude = 550
    max_time = 1

    final_tensor, df_filtered, latitudes_tensor, longitudes_tensor = apply_physical_data(login, df, buf, max_latitude, max_longitude, max_time)
    torch.save(final_tensor, f"tensor_picture_space_{space_buffer}_pfts.pt")
    torch.save(latitudes_tensor, f"tensor_latitude_space_{space_buffer}_pfts.pt")
    torch.save(longitudes_tensor, f"tensor_longitude_space_{space_buffer}_pfts.pt")
    df_filtered.to_csv(f'tensor_df_space_{space_buffer}_pfts.csv')
    
if __name__ == "__main__":
    main()