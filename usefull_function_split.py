#Usefull Function for split

import pandas as pd
from math import radians, sin, cos, sqrt, atan2 

def haversine2(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points 
    on the Earth's surface given their latitude and longitude 
    in decimal degrees.

    Formula: haversine

    Parameters:
    lat1 (float): Latitude of point 1 in radian
    lon1 (float): Longitude of point 1 in radian
    lat2 (float): Latitude of point 2 in radian
    lon2 (float): Longitude of point 2 in radian

    Returns:
    distance (float): Distance between the two points in kilometers
    """    
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c  # Earth radius in kilometers
    return distance


def check_min_distance(df1, df2, min_distance, max_time_diff):
    list_test_to_add = []
    list_train_to_add = []
    sum = 0
    for idx1, row1 in df1.iterrows():
        stop_iteration = False 
        for idx2, row2 in df2.iterrows():
            if stop_iteration:
                break
            distance = haversine2(row1['lat'], row1['lon'], row2['lat'], row2['lon'])
            if distance < min_distance:
                date1 = row1['date']
                date2 = row2['date']
                time_diff = abs((date2 - date1).days)
                if time_diff <= max_time_diff:
                    sum +=1
                    if sum % 2 == 0:
                        list_train_to_add.append(row2)
                        df2.drop(idx2, inplace=True) 
                    else : 
                        list_test_to_add.append(row1)
                        df1.drop(idx1, inplace=True)
                        stop_iteration = True                   

    return sum, df1, df2, list_test_to_add, list_train_to_add

def condition_fit(df_train, df_val, min_distance, max_time_diff): 
    sum = 100

    while sum != 0 : 
        sum, df_train, df_val, list_val_to_add, list_train_to_add = check_min_distance(df_train, df_val, min_distance, max_time_diff)
        print(sum, df_train.shape[0], df_val.shape[0])
        if sum < 10: 
            df_val_add = pd.concat([pd.DataFrame(list_train_to_add), pd.DataFrame(list_val_to_add)], ignore_index=True)
            df_val = pd.concat([df_val, df_val_add], ignore_index=True)
            df_val['test'] = 'ok'
            return df_train, df_val
        else : 
            df_train_to_add = pd.DataFrame(list_train_to_add)
            df_val_to_add = pd.DataFrame(list_val_to_add)
        
            df_train = pd.concat([df_train, df_train_to_add])
            df_val = pd.concat([df_val, df_val_to_add])



from haversine import haversine, Unit
def find_non_compliant_points(train_df, val_df, distance_threshold, date_threshold):
    #train_df['date'] = train_df['date'].str.split().str[0]  # Supprimer la partie "00:00:00" si elle existe
    #val_df['date'] = val_df['date'].str.split().str[0]

    # Convertir les colonnes 'date' en datetime avec un format explicite
    train_df['date'] = pd.to_datetime(train_df['date'])
    train_df['date'] = train_df['date'].dt.date
    val_df['date'] = pd.to_datetime(val_df['date'])
    val_df['date'] = val_df['date'].dt.date

    non_compliant_points = []  # Liste pour stocker les points non conformes
    
    # Itérer sur une copie de val_df pour éviter les erreurs lors de la suppression
    for val_index, val_point in val_df.iterrows():
        val_coords = (val_point['lat'], val_point['lon'])
        val_date = val_point['date']
        
        compliant = False  # Drapeau pour savoir si on a trouvé un point d'entraînement valide
        
        # Itérer sur une copie de train_df pour éviter les erreurs lors de la suppression
        for train_index, train_point in train_df.iterrows():
            train_coords = (train_point['lat'], train_point['lon'])
            train_date = train_point['date']
            
            # Calculer la distance géographique
            geo_distance = haversine(train_coords, val_coords, unit=Unit.KILOMETERS)
            
            # Calculer la différence de date (en jours)
            date_difference = abs((train_date - val_date).days)
            
            # Vérifier si le point de validation est conforme
            if geo_distance <= distance_threshold and date_difference <= date_threshold:
                compliant = True
        
        # Si le point n'est pas conforme, l'ajouter à la liste et le supprimer des DataFrames
        if not compliant:
            non_compliant_points.append(val_point)
            non_compliant_points.append(train_point)  # Ajouter le point non conforme à la liste
            
            # Supprimer les points non conformes de train_df et val_df
            train_df.drop(train_index, inplace=True)
            val_df.drop(val_index, inplace=True)
    
    # Convertir la liste en DataFrame
    non_compliant_df = pd.DataFrame(non_compliant_points)
    
    # Retourner le DataFrame des points non conformes
    return non_compliant_df, train_df, val_df

def find_closest(train_df, val_df):
    min_distance = float('inf')
    closest_train = None
    closest_val = None
    
    # Parcourir chaque point d'entraînement (train)
    for i, train_point in train_df.iterrows():
        train_coords = (train_point['lat'], train_point['lon'])
        train_date = train_point['date']
        
        # Parcourir chaque point de validation (val)
        for j, val_point in val_df.iterrows():
            val_coords = (val_point['lat'], val_point['lon'])
            val_date = val_point['date']
            
            # Calculer la distance géographique avec la formule de Haversine
            geo_distance = haversine(train_coords, val_coords, unit=Unit.KILOMETERS)
            
            # Calculer la différence de date (en jours)
            date_difference = abs((train_date - val_date).days)
            
            # Somme des deux distances (géographique + temporelle)
            total_distance = geo_distance + date_difference
            
            # Trouver la distance minimale
            if total_distance < min_distance:
                min_distance = total_distance
                closest_train = train_point
                closest_val = val_point
    
    return closest_train, closest_val, min_distance
    