import torch 
from geopy.distance import geodesic
from datetime import datetime

def from_date_to_year_month_day(date_str):
    date_obj = datetime.strptime(date_str, '%Y-%m-%d')
    
    year = str(date_obj.year)
    month = str(date_obj.month).zfill(2) 
    day = str(date_obj.day).zfill(2)  

    return year, month, day


def selective_data(dataset, var, lat_range, lon_range, lat_name="latitude", lon_name="longitude"):
    """
    Sélectionne une sous-région dans un dataset xarray et la convertit en tenseur PyTorch (GPU),
    avec les indices de latitude et de longitude correspondants.
    """
    subset = dataset[var].sel(**{
        lat_name: slice(lat_range[1], lat_range[0]),
        lon_name: slice(lon_range[0], lon_range[1]),
    })

    if subset.size == 0:
        subset = dataset[var].sel(**{
            lat_name: dataset[lat_name].sel({lat_name: lat_range[0]}, method="nearest"),
            lon_name: dataset[lon_name].sel({lon_name: lon_range[0]}, method="nearest"),
        })

    if "time" in subset.dims:
        subset = subset.squeeze(dim="time", drop=True)
    
    # Récupération des indices de latitude et longitude
    lat_indices = torch.tensor(dataset[lat_name].values, dtype=torch.float32, device=device)
    lon_indices = torch.tensor(dataset[lon_name].values, dtype=torch.float32, device=device)
    
    selected_lat_indices = lat_indices[(lat_indices >= lat_range[0]) & (lat_indices <= lat_range[1])]
    selected_lon_indices = lon_indices[(lon_indices >= lon_range[0]) & (lon_indices <= lon_range[1])]

    tensor = torch.tensor(subset.values, dtype=torch.float32, device=device)

    return tensor, selected_lat_indices, selected_lon_indices
