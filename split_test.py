import pandas as pd 
import torch
import os
path = "/home/elabourdette/complex/gdrive/shared/Proj_FORMAL/deep_satellite/Code/CNN/pytorch_process/dataset/from_pfts_study_reset12072025/"
df = pd.read_csv(os.path.join(path, 'tensor_df_space_200_study_brewin_done_zone_done_translate_done.csv'))
latitude = torch.load(os.path.join(path, 'tensor_latitude_space_200_maredatlov.pt'))
longitude = torch.load(os.path.join(path, 'tensor_longitude_space_200_maredatlov.pt'))
picture = torch.load(os.path.join(path, 'tensor_picture_space_200_maredatlov.pt'))

latitude = latitude.permute(1, 0, 2)
longitude = longitude.permute(1, 0, 2)

nb_lines = int(longitude.shape[0]/7)
latitude = latitude.reshape(nb_lines,7,100)
longitude = longitude.reshape(nb_lines,7,550)
picture = picture.reshape(nb_lines,18,100,550)

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import hdbscan

# Assurez-vous que votre colonne 'date' est bien au format datetime
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year

# Pour stocker les résultats
all_dfs = []

# Compteur pour les clusters uniques des points non clusterisés
unique_cluster_id = 0

for year, group in df.groupby('year'):
    print(f"== Clustering pour l'année {year} ==")

    # Préparation temporelle locale à l’année
    group = group.copy()
    group['days_since_start'] = (group['date'] - group['date'].min()).dt.days

    # Construction des features spatio-temporelles (pondération temporelle)
    features = np.column_stack([
        group['lat'].values,
        group['lon'].values,
        group['days_since_start'].values / 5  # réduction temporelle
    ])

    # Standardisation
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    # HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=10, metric='euclidean')
    labels = clusterer.fit_predict(features_scaled)

    # Ajout des labels dans le dataframe
    group['cluster'] = labels

    # Attribution d'un cluster unique aux points non clusterisés
    unclustered_points = group[group['cluster'] == -1].copy()
    unclustered_points['cluster'] = [f"{year}_U{unique_cluster_id + i:04d}" for i in range(len(unclustered_points))]
    unique_cluster_id += len(unclustered_points)

    clustered_points = group[group['cluster'] != -1].copy()
    clustered_points['cluster'] = clustered_points['cluster'].apply(lambda c: f"{year}_{int(c):04d}")

    # Concaténation des points clusterisés et non clusterisés
    group = pd.concat([clustered_points, unclustered_points])

    all_dfs.append(group)

# Concaténation finale
df_clustered = pd.concat(all_dfs, ignore_index=True)

# Vérification
print(f"Nombre total de points clusterisés : {len(df_clustered)}")
print(f"Nombre total de clusters : {df_clustered['cluster'].nunique()}")

import pandas as pd
import plotly.express as px


# Liste des années uniques triées
years = sorted(df_clustered['year'].dropna().unique())

# Boucle sur chaque année
for year in years:
    df_year = df_clustered[df_clustered['year'] == year].copy()

    # Création de l'infobulle
    df_year['hover_text'] = df_year.apply(
        lambda row: f"Date: {row['date'].date()}<br>"
                    f"TChla: {row['tchla']}<br>"
                    f"Micro: {row['micro_chla B']}<br>"
                    f"Nano: {row['nano_chla B']}<br>"
                    f"Pico: {row['pico_chla B']}<br>"
                    f"Cluster: {row['cluster']}", axis=1
    )

    # Création de la carte
    fig = px.scatter_geo(
        df_year,
        lat='lat',
        lon='lon',
        color='cluster',
        hover_name='date',
        hover_data={'lat': False, 'lon': False, 'cluster': False, 'hover_text': True},
        title=f"Distribution géographique des points – {year} (par cluster)",
        opacity=0.8
    )

    # Mise en forme de la carte
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(243, 243, 243)",
            projection_type='natural earth'
        ),
        hoverlabel=dict(bgcolor="white", font_size=11),
        width=1000,
        height=600
    )

    fig.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = df_clustered.copy()
df = df.dropna(subset=['tchla'])

train_list, val_list, test_list = [], [], []

def make_bins(series, n_bins=10):
    return pd.qcut(series, q=n_bins, duplicates='drop')

for cluster_id, df_cluster in df.groupby('cluster'):
    df_cluster = df_cluster.copy()
    
    # Binning pour stratification
    df_cluster['tchla_bin'] = make_bins(df_cluster['tchla'])

    # Vérifier si assez de classes (>1) pour stratifier
    try:
        df_train, df_temp = train_test_split(
            df_cluster,
            test_size=0.2,
            stratify=df_cluster['tchla_bin'],
            random_state=42
        )
    except ValueError:
        # Pas assez d'échantillons pour stratifier, faire split sans stratify
        df_train, df_temp = train_test_split(
            df_cluster,
            test_size=0.2,
            random_state=42
        )

    # Même logique pour test/val
    try:
        df_val, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            stratify=df_temp['tchla_bin'],
            random_state=42
        )
    except ValueError:
        df_val, df_test = train_test_split(
            df_temp,
            test_size=0.5,
            random_state=42
        )

    train_list.append(df_train)
    val_list.append(df_val)
    test_list.append(df_test)

# Fusion finale
df_train = pd.concat(train_list).drop(columns='tchla_bin')
df_val = pd.concat(val_list).drop(columns='tchla_bin')
df_test = pd.concat(test_list).drop(columns='tchla_bin')

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ajouter une colonne log10(tchla) dans chaque split (filtrée pour valeurs > 0)
df_train['log10_tchla'] = np.log10(df_train['tchla'].clip(lower=1e-6))
df_val['log10_tchla'] = np.log10(df_val['tchla'].clip(lower=1e-6))
df_test['log10_tchla'] = np.log10(df_test['tchla'].clip(lower=1e-6))

# Tracé
plt.figure(figsize=(10, 6))
sns.kdeplot(df_train['log10_tchla'], label='Train', fill=True, alpha=0.4)
sns.kdeplot(df_val['log10_tchla'], label='Validation', fill=True, alpha=0.4)
sns.kdeplot(df_test['log10_tchla'], label='Test', fill=True, alpha=0.4)

plt.xlabel("log10(tchla)")
plt.ylabel("Density")
plt.title("Distribution de log10(tchla) par split")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


import pandas as pd
import plotly.express as px

# Liste des années uniques triées
years = sorted(df_all['year'].dropna().unique())

# Boucle sur chaque année
for year in years:
    df_year = df_all[df_all['year'] == year].copy()

    # Création de l'infobulle
    df_year['hover_text'] = df_year.apply(
        lambda row: f"Date: {row['date'].date()}<br>"
                    f"TChla: {row['tchla']}<br>"
                    f"Micro: {row['micro_chla B']}<br>"
                    f"Nano: {row['nano_chla B']}<br>"
                    f"Pico: {row['pico_chla B']}<br>"
                    f"Cluster: {row['cluster']}", axis=1
    )

    # Carte avec couleurs par split
    fig = px.scatter_geo(
        df_year,
        lat='lat',
        lon='lon',
        color='split',  # <- Couleur par ensemble
        hover_name='date',
        hover_data={'lat': False, 'lon': False, 'cluster': False, 'hover_text': True},
        title=f"Distribution géographique des points – {year} (par split)",
        opacity=0.8,
        category_orders={'split': ['train', 'val', 'test']}  # Pour ordre fixe
    )

    # Mise en forme
    fig.update_traces(marker=dict(size=6))
    fig.update_layout(
        geo=dict(
            showland=True,
            landcolor="rgb(243, 243, 243)",
            projection_type='natural earth'
        ),
        hoverlabel=dict(bgcolor="white", font_size=11),
        width=1000,
        height=600
    )

    fig.show()

import matplotlib.pyplot as plt

# Préparer la figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Définir les splits
splits = ['train', 'val', 'test']
colors = plt.cm.tab20.colors  # Palette de couleurs

# Boucle sur chaque split
for i, split in enumerate(splits):
    df_split = df_all[df_all['split'] == split]
    zone_counts = df_split['zone'].value_counts()
    labels = zone_counts.index
    sizes = zone_counts.values

    axes[i].pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=140,
        colors=colors[:len(labels)],
        textprops={'fontsize': 9}
    )
    axes[i].axis('equal')
    axes[i].set_title(f'{split.upper()} - Répartition des zones')

# Titre global
plt.suptitle('Répartition des zones par split (train / val / test)', fontsize=14)
plt.tight_layout()
plt.subplots_adjust(top=0.85)
plt.show()

