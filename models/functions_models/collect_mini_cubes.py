import torch 
from tqdm import tqdm
import torch.nn.functional as F
from .get_lon_lat_bounds_geodesic import * 

#######################################################
###################  MAIN FUNCTION  ###################
#######################################################

def collect_mini_cubes(df_tensor, final_tensor_test, latitudes_tensor_test, longitudes_tensor_test, space_buffer,max_latitude=100, max_longitude=100, device="cpu"):
    tensor_mini_cubes = []
    total_rows = df_tensor.shape[0]
    #max_latitude = int((space_buffer * 100) / 200)
    #max_longitude = int((space_buffer * 500) / 200)

    
    df_tensor = df_tensor.to(device)
    index_to_drop = []
    for index in tqdm(range(total_rows), total=total_rows, desc="Processing Data", leave=False):
        lat, lon = df_tensor[index, 0], df_tensor[index, 1]
        buf = [space_buffer, space_buffer]
        lat_inter, lon_inter = get_lon_lat_bounds_geodesic(lat, lon, buf)
        
        mini_cube_all_data = []  
        sum_nan = 0 
        for variable in range(final_tensor_test.shape[1]):
            tensor = final_tensor_test[index, variable, :, :].to(device)
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
    
            if variable < 4:
                index_lat_lon = 0
                method = 'other'
            elif variable == 4:
                index_lat_lon = 1
                method = 'other'
            elif variable == 5:
                index_lat_lon = 2
                method = 'other'
            elif variable == 6:
                index_lat_lon = 3
                method = 'fsle'
            elif variable < 13:
                index_lat_lon = 4
                method = 'other'
            else:
                index_lat_lon = 5
                method = 'other'
    
            latitude_values = latitudes_tensor_test[index, index_lat_lon, :].to(device)
            longitude_values = longitudes_tensor_test[index, index_lat_lon, :].to(device)
    
            mini_cube = extract_mini_cubes(tensor, latitude_values, longitude_values, lat_inter[0], lon_inter[0], method)
            
            if mini_cube.shape[1] == 0 :
                index_to_drop.append(index)
                sum_nan +=1
                break
            else : 
                mini_cube_reshape = reshape_all_same(mini_cube, max_latitude, max_longitude, device=device)
                mini_cube_all_data.append(mini_cube_reshape)
        
        
        if sum_nan == 0 :
            mini_cube_all_data = torch.stack(mini_cube_all_data) 
            tensor_mini_cubes.append(mini_cube_all_data)
    
    tensor_mini_cubes = torch.stack(tensor_mini_cubes)
    df_mini_cubes = drop_rows(df_tensor, index_to_drop, device)
    
    return tensor_mini_cubes.to(device) , df_mini_cubes.to(device)


#######################################################
###################  SUB FUNCTIONS  ###################
#######################################################

###################  EXTRACT MINI CUBES  ###################

def extract_mini_cubes(tensor, selected_lat_indices, selected_lon_indices, mini_lat_range, mini_lon_range, method='regular', device='cpu'):
    """
    Récupère un mini cube à l'intérieur du grand tensor en fonction d'un nouvel intervalle de latitude et de longitude.
    """
    # Convertir les bornes en tenseurs scalaires sur le bon device
    mini_lat_min = torch.tensor(mini_lat_range[1], dtype=selected_lat_indices.dtype, device=device)
    mini_lat_max = torch.tensor(mini_lat_range[0], dtype=selected_lat_indices.dtype, device=device)
    mini_lon_min = torch.tensor(mini_lon_range[0], dtype=selected_lon_indices.dtype, device=device)
    mini_lon_max = torch.tensor(mini_lon_range[1], dtype=selected_lon_indices.dtype, device=device)

    # S'assurer que selected_lat_indices et selected_lon_indices sont bien sur le bon device
    selected_lat_indices = selected_lat_indices.to(device)
    selected_lon_indices = selected_lon_indices.to(device)

    # Trouver les indices qui respectent la plage définie
    if method == 'optic':
        mini_lat_indices = torch.nonzero((selected_lat_indices <= mini_lat_max) & (selected_lat_indices >= mini_lat_min)).squeeze()
        mini_lon_indices = torch.nonzero((selected_lon_indices >= mini_lon_min) & (selected_lon_indices <= mini_lon_max)).squeeze()

    elif method == 'fsle': 
        mini_lon_min = (mini_lon_min + 360) % 360 if mini_lon_min < 0 else mini_lon_min
        mini_lon_max = (mini_lon_max + 360) % 360 if mini_lon_max < 0 else mini_lon_max
        mini_lat_indices = torch.nonzero((selected_lat_indices <= mini_lat_min) & (selected_lat_indices >= mini_lat_max)).squeeze()
        mini_lon_indices = torch.nonzero((selected_lon_indices >= mini_lon_min) & (selected_lon_indices <= mini_lon_max)).squeeze()

    else:
        mini_lat_indices = torch.nonzero((selected_lat_indices <= mini_lat_min) & (selected_lat_indices >= mini_lat_max)).squeeze()
        mini_lon_indices = torch.nonzero((selected_lon_indices >= mini_lon_min) & (selected_lon_indices <= mini_lon_max)).squeeze()
    
    # Gérer les cas où on obtient un seul élément
    mini_lat_indices = mini_lat_indices.tolist() if mini_lat_indices.dim() > 0 else [mini_lat_indices.item()]
    mini_lon_indices = mini_lon_indices.tolist() if mini_lon_indices.dim() > 0 else [mini_lon_indices.item()]

    # Extraire le mini cube du tensor et s'assurer qu'il est sur le bon device
    mini_cube = tensor.to(device)[mini_lat_indices, :][:, mini_lon_indices]

    return mini_cube

###################  RESHAPE_ALL_SAME  ###################
def reshape_all_same(subset, lat_max, lon_max, device="cpu"):
    subset = subset.to(device)  
    if subset.ndimension() == 0: 
        return subset.expand(lat_max, lon_max)
    
    if subset.ndimension() == 1: 
        subset = subset.unsqueeze(1)
    
    lat, lon = subset.shape  
    
    subset = subset.unsqueeze(0).unsqueeze(0)  
    #subset_resized = F.interpolate(subset, size=(lat_max, lon_max), mode='nearest')
    subset_resized = F.interpolate(subset, size=(lat_max, lon_max), mode='bilinear')
    
    return subset_resized.squeeze(0).squeeze(0) 

###################  COUNT_THRESHOLD_NAN  ###################

def count_threshold_nan(tensor_mini_cubes, df_tensor, threshold=0.5, device='cpu'):

    tensor_mini_cubes = tensor_mini_cubes.to(device)
    df_tensor = df_tensor.to(device)

    num_images, num_variables, _, _ = tensor_mini_cubes.shape
    valid_indices = []

    for idx in range(num_images):
        keep_image = True 
        
        for var in range(num_variables): 
            nan_mask = torch.isnan(tensor_mini_cubes[idx, var, :, :])
            nan_proportion = torch.sum(nan_mask).item() / tensor_mini_cubes[idx, var, :, :].numel()

            if nan_proportion > threshold:  
                keep_image = False
                break  
            elif var == 18 : 
                bathy = tensor_mini_cubes[idx, var, :, :]
        
        if keep_image:
            valid_indices.append(idx)

    # Filtrage des données
    valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long, device=device)
    tensor_mini_cubes_no_nan = tensor_mini_cubes[valid_indices_tensor, :, :, :]
    df_no_nan = df_tensor[valid_indices_tensor, :]

    return tensor_mini_cubes_no_nan, df_no_nan

###################  DROP_ROWS ###################
def drop_rows(df_tensor, index_to_drop, device='cpu'):

    mask = torch.ones(df_tensor.shape[0], dtype=torch.bool)
    mask[index_to_drop] = False
    
    df_tensor = df_tensor[mask].to(device)
    
    return df_tensor