import numpy as np
import pandas as pd
import plotly.express as px
from matplotlib import pyplot as plt
from plotnine import *
import json 

def Zone(row):
    lat  = row['lat']
    lon  = row['lon'] 
    zone = 'OTHER'
    
    if lat>=0 :     
        if (lat < 10):
            zone = 'PEQ (Positif Equator)'
        elif (lat < 40):
            zone = 'PSubTG (Positif SubTropical Gyre)'
        elif (lat < 60):
            zone = 'PSubP (Positif SubPolar Gyre)'
        else : 
            zone = 'PHL (Positif High Latitude)'
            
        if ((lat > 35) & (lat < 44) & (lon > -5.5) & (lon < 6)):
            zone = 'MS (Mediterranean Sea)'
    
        if ((lat > 35) & (lat < 45) & (lon > 5) & (lon < 15.5)):
            zone = 'MS (Mediterranean Sea)'
        
        if ((lat > 30) & (lat < 45) & (lon > 15.5) & (lon < 25.5)):
            zone = 'MS (Mediterranean Sea)'
        
        if ((lat > 30) & (lat < 40) & (lon > 22.5) & (lon < 36)):
            zone = 'MS (Mediterranean Sea)'
        
        if ((lat > 41) & (lat < 46) & (lon > 12) & (lon < 15.5)):
            zone = 'MS (Mediterranean Sea)'
    
        if ((lat > 30) & (lat < 37.5) & (lon > 10) & (lon < 15.5)):
            zone = 'MS (Mediterranean Sea)'
    else : 
        lat = -lat
        if (lat < 10):
            zone = 'NEQ (Negatif Equator)'
        elif (lat < 40):
            zone = 'NSubTG (Negatif SubTropical Gyre)'
        elif (lat < 60):
            zone = 'NSubP (Negatif SubPolar Gyre)'
        else : 
            zone = 'NHL (Negatif High Latitude)'

        
    
    return zone

def determine_ocean(row):
    lat  = row['lat']
    lon  = row['lon'] 
    zone = 'Other'
    
    if (lat < -40):
        zone = 'AnW (Antartic Waters)'
    if ((lat > 65) & (lat < 90) & (lon > -12.5) & (lon < 20)):
        zone = 'AtW (Atlantic Waters)'
    if ((lat > 17.5) & (lat < 32.5) & (lon > -85) & (lon < -15.5)):
        zone = 'AtW (Atlantic Waters)'
    if ((lat >= 0) & (lat < 60) & (lon > -180) & (lon < -100)):
        zone = 'PW (Pacific Waters)'
    if ((lat >= 0) & (lat < 17.5) & (lon > -65) & (lon < 11)):
        zone = 'AtW (Atlantic Waters)'
    if ((lat > -50) & (lat < 0) & (lon > -180) & (lon < -70)):
        zone = 'PW (Pacific Waters)'
    if ((lat > -30) & (lat < 0) & (lon > -45) & (lon < 13)):
        zone = 'AtW (Atlantic Waters)'
    if ((lat > -45) & (lat < -30) & (lon > -58) & (lon < 20)):
        zone = 'AtW (Atlantic Waters)'
    if ((lat > -45) & (lat < 0) & (lon > 45) & (lon < 110)):
        zone = 'IW (Indian Waters)'
    if ((lat > -14.1) & (lat < 25) & (lon > 40) & (lon < 115)):
        zone = 'IW (Indian Waters)'
    if ((lat > 35) & (lat < 44) & (lon > -5.5) & (lon < 6)):
        zone = 'MS (Mediterranean Sea)'
    if ((lat > 35) & (lat < 45) & (lon > 5) & (lon < 15.5)):
        zone = 'MS (Mediterranean Sea)'
    if ((lat > 30) & (lat < 45) & (lon > 15.5) & (lon < 25.5)):
        zone = 'MS (Mediterranean Sea)'
    if ((lat > 30) & (lat < 40) & (lon > 22.5) & (lon < 36)):
        zone = 'MS (Mediterranean Sea)'
    if ((lat > 41) & (lat < 46) & (lon > 12) & (lon < 15.5)):
        zone = 'MS (Mediterranean Sea)'
    if ((lat > 30) & (lat < 37.5) & (lon > 10) & (lon < 15.5)):
        zone = 'MS (Mediterranean Sea)'
    if ((lat > 40.98) & (lat < 47.35) & (lon > 27.67) & (lon < 42.5)):
        zone = 'Other'#'BKS (Black Sea)'
    if ((lat > 10.78) & (lat < 30.86) & (lon > 32.78) & (lon < 42.4)):
        zone = 'Other'#'RDS (Red Sea)'
    if ((lat > -45) & (lat < -11.4) & (lon > 109) & (lon < 114)):
        zone = 'IW (Indian Waters)'
    if ((lat >= 0) & (lat < 11) & (lon > -95) & (lon < -78)):
        zone = 'PW (Pacific Waters)'
    if ((lat < 65) & (lat > 30) & (lon < -5.5) & (lon > -80)):
        zone = 'AtW (Atlantic Waters)'
    if (lat > 65):
        zone = 'AW (Artic Water)'
    if ((lat < 60) & (lat > 0) & (lon > 140) & (lon < 180)):
        zone = 'PW (Pacific Waters)'
    if ((lat < 0) & (lat > -40) & (lon > 140) & (lon < 180)):
        zone = 'PW (Pacific Waters)'
    
    if ((zone == 'OTHER') & (lat > -30) & (lat < -25) & (lon > 0) & (lon < 50)):
        zone = 'IW (Indian Waters)'
    return zone

def determine_ocean_precise(row):
    lat  = row['lat']
    lon  = row['lon'] 
    zone = 'OTHER'
    
    if (lat < -40):
        zone = 'AW (Antartic Waters)'
    if ((lat >= 40) & (lon< -100)):
        zone = 'NPSPG (North Pacific SubPolar Gyre)'
    if ((lat > 60) & (lat < 80) & (lon > -12.5) & (lon < 20)):
        zone = 'NS (Northern Seas (Groenland, Barents and Norway seas))'
    if ((lat > 17.5) & (lat < 32.5) & (lon > -85) & (lon < -15.5)):
        zone = 'NASTG (North Atlantic SubTropical Gyre)'
    if ((lat > 17.5) & (lat < 33) & (lon > -175) & (lon < -140)):
        zone = 'NPSTG (North Pacific SubTropical Gyre)'
    if ((lat > -33) & (lat < -17) & (lon > -180) & (lon < -84)):
        zone = 'SPSTG (South Pacific SubTropical Gyre)'
    if ((lat > -19) & (lat < 6) & (lon > -125) & (lon < -73)):
        zone = 'UPW (Upwelling)'
    if ((lat >= 0) & (lat < 17.5) & (lon > -35) & (lon < 11)):
        zone = 'NAEW (North Atlantic Equatorial Waters)'
    if ((lat > -23) & (lat < 23) & (lon > -160) & (lon < -90)):
        zone = 'PEW (Pacific Equatorial Waters)'
    if ((lat > -30) & (lat < 0) & (lon > -45) & (lon < 11)):
        zone = 'SAEC (South Atlantic Equatorial Waters)'
    if ((lat > -45) & (lat < -30) & (lon > -58) & (lon < 20)):
        zone = 'SASTG (South Atlantic SubTropical Gyre)'
    if ((lat > -45) & (lat < -15) & (lon > 45) & (lon < 110)):
        zone = 'SISTG (South Indian SubTropical Gyre)'
    if ((lat > -14.1) & (lat < 7.5) & (lon > 40) & (lon < 115)):
        zone = 'IEQ (Indian Equatorial Waters)'
    if ((lat > 7.5) & (lat < 25) & (lon > 43) & (lon < 100)):
        zone = 'IOMZ (Indian Oxygen Minimum Zones (Arabian Sea and Bengal Bay))'
    if ((lat > 35) & (lat < 44) & (lon > -5.5) & (lon < 6)):
        zone = 'WMS (Western Mediterranean Sea)'
    if ((lat > 35) & (lat < 45) & (lon > 5) & (lon < 15.5)):
        zone = 'WMS (Western Mediterranean Sea)'
    if ((lat > 30) & (lat < 45) & (lon > 15.5) & (lon < 25.5)):
        zone = 'EMS (Eastern Mediterranean Sea)'
    if ((lat > 30) & (lat < 40) & (lon > 22.5) & (lon < 36)):
        zone = 'EMS (Eastern Mediterranean Sea)'
    if ((lat > 41) & (lat < 46) & (lon > 12) & (lon < 15.5)):
        zone = 'EMS (Eastern Mediterranean Sea)'
    if ((lat > 30) & (lat < 37.5) & (lon > 10) & (lon < 15.5)):
        zone = 'EMS (Eastern Mediterranean Sea)'
    if ((lat > 40.98) & (lat < 47.35) & (lon > 27.67) & (lon < 42.5)):
        zone = 'BKS (Black Sea)'
    if ((lat > 10.78) & (lat < 30.86) & (lon > 32.78) & (lon < 42.4)):
        zone = 'RDS (Red Sea)'
    if ((lat > -40) & (lat < -11.4) & (lon > 109) & (lon < 114)):
        zone = 'AUS (Australia Waters)'
    if ((lat > 0) & (lat < 24) & (lon > 100) & (lon < 120)):
        zone = 'CHINS (Chineese Sea)'
    if ((lat > -30) & (lat < -5) & (lon > 142) & (lon < 180)):
        zone = 'ARCH (Archipelagos waters)'
    if ((lat > 22) & (lat < 36.10) & (lon > -135) & (lon < -119)):
        zone = 'CAL (California current)'
    if ((lat > 0) & (lat < 22) & (lon > -165) & (lon < -111)):
        zone = 'PEW (Pacific Equatorial Waters)' 
    if ((lat < 65) & (lat > 30) & (lon < -5.5) & (lon > -80)):
        zone = 'NA (North Atlantic waters)'
    if (lat > 65):
        zone = 'AW (Artic Water)'
    if ((lat < 49) & (lat > 35) & (lon > 140) & (lon < 170)):
        zone = 'PSPG (Pacific SubPolar Gyre) '
    if ((lat < 48) & (lat > 35) & (lon > -130) & (lon < -123)):
        zone = 'PSPG (Pacific SubPolar Gyre)'
    if ((lat > 65) & (lat < 70) & (lon > -72) & (lon < -50)):
        zone = 'BAFF (Baffin Bay)'
    if ((lat > 70) & (lat < 80) & (lon > -80) & (lon < -50)):
        zone = 'BAFF (Baffin Bay)'
    if ((zone == 'OTHER') & (lat > -30) & (lat < -25) & (lon > 0) & (lon < 50)):
        zone = 'STZ (Sub Tropical Zone)'
    # add Peru upwelling
    if ((lat > -36) & (lat < -33)  & (lon > -82) & (lon < -72)):
        zone = 'UPW (Upwelling)'
    # add Namibia 
    if ((lat > -31) & (lat < -21) & (lon > 12) & (lon < 15)):
        zone = 'UPW (Upwelling)'
    # add California 
    if ((lat > 22.3) & (lat < 29) & (lon > -116) & (lon < -111)):
        zone = 'UPW (Upwelling)'
    return zone


def define_waters_zones(space_buffer, date_buffer, bathy): 
    # access token for the Mapbox tile service (for plotting)
    mapbox_token = 'pk.eyJ1Ijoiam9pcmlzc29uIiwiYSI6ImNsdTR0amt2NTFnd3AybG4wa3BncjJkdTYifQ.Av6HvdSFlDKvOZZJ642UZg'
    # custom style with dark oceans and lighter land
    my_mapbox_style = 'mapbox://styles/joirisson/clu4uxvra00b201p6cqky68t9'
    # my_mapbox_style = 'carto-positron'

    data = pd.read_csv(f'/home/elabourdette/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/chla_model-copernicusmarine/machine_learning_files/s_{space_buffer}_d_{date_buffer}/bathy_{bathy}/data_completed_s_{space_buffer}_d_{date_buffer}.csv')
    
    
    data['zone'] = data.apply(Zone, axis=1)
    data['ocean'] = data.apply(determine_ocean, axis=1)
    data = data.loc[:, ~data.columns.str.startswith('Unnamed')]
    data.to_csv(f'/home/elabourdette/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/chla_model-copernicusmarine/machine_learning_files/s_{space_buffer}_d_{date_buffer}/bathy_{bathy}/data_completed_s_{space_buffer}_d_{date_buffer}.csv',index=False)

def main(): 
    date_buffer = 1
    bathy = 500 
    for space_buffer in [16]:
        define_waters_zones(space_buffer, date_buffer, bathy)

if __name__ == "__main__":
    main()