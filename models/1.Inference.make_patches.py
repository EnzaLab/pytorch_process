# Generate patches for each date 
import os
import joblib
import datetime 
import calendar
import pandas as pd 
import xarray as xr
from functions_models.extract_big_cubes_CPU import *

#Make it one time #
def generate_patches(min_lat, max_lat, min_lon, max_lon, patch_size):
    lat_points = np.arange(min_lat, max_lat, patch_size)
    lon_points = np.arange(min_lon, max_lon, patch_size)

    data = {
        "Lat": [],
        "Lon": []
    }

    for lat in lat_points:
        for lon in lon_points:
            data["Lat"].append(lat)
            data["Lon"].append(lon)

    return pd.DataFrame(data)
    
def make_patches(login, spatial_resolution):
    spatial_resolution = 0.035 #in degree
    patches = generate_patches(min_lat=-80, max_lat=80, min_lon=-180, max_lon=180, patch_size=spatial_resolution)
    
    interpolator = joblib.load(f'{login}/complex/share/save_training_Enza/interpolator_bathy.pkl')
    
    points = np.column_stack((patches['Lat'].values, patches['Lon'].values))
    
    bathy = interpolator(points)
    patches['Bathy'] = bathy
    
    patches_bathy_ocean = patches[patches['Bathy']<-500]
    patches_bathy_ocean.to_csv('./patches_inference.csv')

def apply_physical_data(login, df, buf, max_latitude, max_longitude, max_time, date): 
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
    rows_complete = []

    for index, row in tqdm(df.iterrows(), total=len(df), desc="Traitement des données"):
        lat = row['Lat']
        lon = row['Lon']
        lat_inter, lon_inter = point_space_boundaries(lat, lon, buf)
        year, month, day = from_date_to_year_month_day(date)

        try:
            path_rrs_completed = os.path.join(path_rrs, f"{year}/{month}/{year}{month}{day}_c3s_obs*.nc")
            path_plk_completed = os.path.join(path_plk, f"{year}/{month}/{year}{month}{day}_c3s_obs*.nc")
            ds_rrs = safe_open_nc(path_rrs_completed, "RRS", date)
            ds_plk = safe_open_nc(path_plk_completed, "PLK", date)
        except Exception:
            rows_complete.append(False)
            continue

        try:
            path_sla_completed = os.path.join(path_sla, f"{year}/{month}/dt_global_allsat_phy_l4_{year}{month}{day}*.nc")
            ds_sla = safe_open_nc(path_sla_completed, "SLA", date)
        except Exception:
            rows_complete.append(False)
            continue

        try:
            path_sst_completed = os.path.join(path_sst, f"{year}/{month}/{year}{month}{day}*.nc")
            ds_sst = safe_open_nc(path_sst_completed, "SST", date)
        except Exception:
            rows_complete.append(False)
            continue

        try:
            path_par_completed = os.path.join(path_par, f"{year}/{month}/{day}/L3m_{year}{month}{day}*.nc")
            ds_par = safe_open_nc(path_par_completed, "PAR", date)
        except Exception:
            rows_complete.append(False)
            continue

        try:
            ds_bathy = xr.open_dataset(path_bathy_completed)
        except Exception:
            rows_complete.append(False)
            continue

        try:
            path_fsle_completed = os.path.join(path_fsle, f"{year}/dt_global_allsat_madt_fsle_{year}{month}{day}*.nc")
            ds_fsle = safe_open_nc(path_fsle_completed, "FSLE", date, login=login)
        except Exception:
            rows_complete.append(False)
            continue

        # Extraction des données (supposé que les fonctions existent déjà)
        RRS412_subset, RRS_lat, RRS_lon = selective_data(ds_rrs, "RRS412", lat_inter[0], lon_inter[0])
        RRS443_subset, _, _ = selective_data(ds_rrs, "RRS443", lat_inter[0], lon_inter[0])
        RRS490_subset, _, _ = selective_data(ds_rrs, "RRS490", lat_inter[0], lon_inter[0])
        RRS510_subset, _, _ = selective_data(ds_rrs, "RRS510", lat_inter[0], lon_inter[0])
        RRS560_subset, _, _ = selective_data(ds_rrs, "RRS560", lat_inter[0], lon_inter[0])
        RRS665_subset, _, _ = selective_data(ds_rrs, "RRS665", lat_inter[0], lon_inter[0])

        CHL_subset, CHL_lat, CHL_lon = selective_data(ds_plk, "CHL", lat_inter[0], lon_inter[0])
        MICRO_subset, _, _ = selective_data(ds_plk, "MICRO", lat_inter[0], lon_inter[0])
        NANO_subset, _, _ = selective_data(ds_plk, "NANO", lat_inter[0], lon_inter[0])
        PICO_subset, _, _ = selective_data(ds_plk, "PICO", lat_inter[0], lon_inter[0])

        sla_subset, SLA_lat, SLA_lon = selective_data(ds_sla, "sla", lat_inter[0], lon_inter[0], method="inverse")
        adt_subset, _, _ = selective_data(ds_sla, "adt", lat_inter[0], lon_inter[0], method="inverse")
        ugos_subset, _, _ = selective_data(ds_sla, "ugos", lat_inter[0], lon_inter[0], method="inverse")
        vgos_subset, _, _ = selective_data(ds_sla, "vgos", lat_inter[0], lon_inter[0], method="inverse")

        sst_subset, SST_lat, SST_lon = selective_data(ds_sst, "analysed_sst", lat_inter[0], lon_inter[0], "lat", "lon", method="inverse")
        par_subset, PAR_lat, PAR_lon = selective_data(ds_par, "PAR_mean", lat_inter[0], lon_inter[0], "lat", "lon")
        fsle_subset, FSLE_lat, FSLE_lon = selective_data(ds_fsle, "fsle_max", lat_inter[0], lon_inter[0], "lat", "lon", method="fsle") 

        bathy_subset, BATHY_lat, BATHY_lon = selective_data(ds_bathy, 'elevation', lat_inter[0], lon_inter[0], "lat", "lon", method="inverse")
        if bathy_subset.size == 1: 
            rows_complete.append(False)
            continue
        else : 
            bathy_subset = bathy_subset[::10,::10]
            BATHY_lat = BATHY_lat[::10]
            BATHY_lon = BATHY_lon[::10]

        rows_complete.append(True)

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
        padded_data = [pad_to_max_size(d, max_latitude, max_longitude) for d in [
            sla_subset, adt_subset, ugos_subset, vgos_subset,
            sst_subset, par_subset, fsle_subset,
            RRS412_subset, RRS443_subset, RRS490_subset, RRS510_subset,
            RRS560_subset, RRS665_subset,
            CHL_subset, MICRO_subset, NANO_subset, PICO_subset,
            bathy_subset
        ]]

        data_tensor = torch.stack(padded_data, dim=0)
        tensor_list.append(data_tensor)

    df_filtered = df[rows_complete].reset_index(drop=True)
    final_tensor = torch.cat(tensor_list, dim=0)

    latitudes_tensor = torch.stack(latitudes_list, dim=0).permute(1, 0, 2)
    longitudes_tensor = torch.stack(longitudes_list, dim=0).permute(1, 0, 2)
        
    final_tensor_test = final_tensor.reshape(df_filtered.shape[0], 18, max_time, max_latitude, max_longitude)
    latitudes_tensor_test = latitudes_tensor.reshape(df_filtered.shape[0], 7, max_latitude)
    longitudes_tensor_test = longitudes_tensor.reshape(df_filtered.shape[0], 7, max_longitude)

    return final_tensor, df_filtered, latitudes_tensor, longitudes_tensor

def main(): 
    login = "/home/elabourdette/"
    space_buffer = 8
    buf = [space_buffer, space_buffer]
    max_latitude = 50
    max_longitude = 150
    max_time = 1
    date = '2002-05-01'
    start_date = datetime.strptime(date, "%Y-%m-%d")
    year = start_date.year
    month = start_date.month
    num_days = calendar.monthrange(year, month)[1]
    patches_bathy_ocean = pd.read_csv('patches_inference.csv')
    #num_days + 1
    for day in tqdm(range(26, 32), desc=f"Processing {year}-{month:02}"):
        date = f"{year}-{month:02d}-{day:02d}"
        day_str = str(day)
  
        final_tensor, df_filtered, latitudes_tensor, longitudes_tensor= apply_physical_data(login, patches_bathy_ocean, buf, max_latitude, max_longitude, max_time,date)
        
        nb_variables = 18
        nb_latitudes = 7 
        shape_0 = int(final_tensor.shape[0]/18)
        latitudes_tensor_reshape = latitudes_tensor.reshape(shape_0, nb_latitudes, max_latitude)
        longitudes_tensor_reshape = longitudes_tensor.reshape(shape_0, nb_latitudes, max_longitude)
        final_tensor_reshape = final_tensor.reshape(shape_0,nb_variables,max_latitude,max_longitude)
        path_to_save = f'{login}/complex/share/save_training_Enza/inference/{year}/{month}/input/{day_str}/'
        os.makedirs(path_to_save, exist_ok=True)

        tensor = torch.tensor(df_filtered[['Lat', 'Lon']].values, dtype=torch.float32)
        torch.save(tensor, os.path.join(path_to_save, f"tensor_latlon_{year}{month}{day_str}.pt"))

        torch.save(final_tensor_reshape, os.path.join(path_to_save, f"tensor_picture_space_{space_buffer}_{year}{month}{day_str}.pt"))
        torch.save(latitudes_tensor_reshape, os.path.join(path_to_save, f"tensor_latitude_space_{space_buffer}_{year}{month}{day_str}.pt"))
        torch.save(longitudes_tensor_reshape, os.path.join(path_to_save, f"tensor_longitude_space_{space_buffer}_{year}{month}{day_str}.pt"))
        df_filtered.to_csv(os.path.join(path_to_save, f'tensor_df_space_{space_buffer}_{year}{month}{day_str}.csv'))
    
        del final_tensor, df_filtered, latitudes_tensor, longitudes_tensor
        del tensor, final_tensor_reshape, latitudes_tensor_reshape, longitudes_tensor_reshape
        torch.cuda.empty_cache() 

if __name__ == "__main__":
    main()