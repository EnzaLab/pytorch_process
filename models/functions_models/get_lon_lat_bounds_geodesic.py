from geopy.distance import geodesic

def get_lon_lat_bounds_geodesic(lat, lon, buffer_km):
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