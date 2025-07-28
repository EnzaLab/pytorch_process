def point_space_boundaries_cpu(lat, lon, buf, device): 
    
    from geopy.distance import geodesic
    
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

import torch

def haversine_distance(lat, lon, distance_km, bearing, device="cuda"):
    # Rayon moyen de la Terre en km
    R = 6371.0
    
    # Conversion en radians
    lat_rad = torch.deg2rad(lat.to(device))
    lon_rad = torch.deg2rad(lon.to(device))
    bearing_rad = torch.deg2rad(bearing.to(device))
    
    d_ratio = torch.tensor(distance_km / R, device=device, dtype=torch.float32)
    new_lat_rad = torch.asin(torch.sin(lat_rad) * torch.cos(d_ratio) +
                             torch.cos(lat_rad) * torch.sin(d_ratio) * torch.cos(bearing_rad))
    new_lon_rad = lon_rad + torch.atan2(torch.sin(bearing_rad) * torch.sin(d_ratio) * torch.cos(lat_rad),
                                        torch.cos(d_ratio) - torch.sin(lat_rad) * torch.sin(new_lat_rad))
    
    return torch.rad2deg(new_lat_rad).to(device), torch.rad2deg(new_lon_rad).to(device)

def point_space_boundaries(lat, lon, buf, device="cuda"):
    lat, lon, buf = lat.to(device), lon.to(device), torch.tensor(buf, device=device)
    
    # Déplacements selon les directions principales
    lat_r, _ = haversine_distance(lat, lon, buf[0], torch.tensor(0.0, device=device), device)
    lat_l, _ = haversine_distance(lat, lon, buf[1], torch.tensor(180.0, device=device), device)
    _, lon_u = haversine_distance(lat, lon, buf[0], torch.tensor(90.0, device=device), device)
    _, lon_d = haversine_distance(lat, lon, buf[1], torch.tensor(-90.0, device=device), device)
    
    tolerance = 0.1
    if lon_d > lon_u:
        if lon_u + 179.9999 < tolerance:
            lon_list = [[lon_d, 179.9999]]
            lat_list = [[lat_l, lat_r]]
        elif lon_d - 179.9999 < tolerance:
            lon_list = [[-179.9999, lon_u]]
            lat_list = [[lat_l, lat_r]]
        else:
            lon_list = [[lon_d, 179.9999], [-179.9999, lon_u]]
            lat_list = [[lat_l, lat_r], [lat_l, lat_r]]
    else:
        lat_list = [[lat_l, lat_r]]
        lon_list = [[lon_d, lon_u]]
    
    return torch.tensor(lat_list, device=device), torch.tensor(lon_list, device=device)
