import matplotlib.pyplot as plt
import numpy as np
import torch
import pandas as pd
import numpy as np 
from tqdm import tqdm
import os
import yaml
import time
import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F


def destandardize(x_stand , mean, std): 
    data = (x_stand * std) + mean
    return data
    
def from_transfo_to_nn(df_train, factor, mean, std):
    df_train = pd.to_numeric(df_train, errors='coerce')
    inv_factor = 1/float(factor)
    df_train = df_train*inv_factor
    df_train = destandardize(df_train, mean, std)
    df_train = np.power(10, df_train)
    
    return df_train

def function_r2(prediction, true, score=False) : 
    ## prediction and true are df column
    prediction = np.array(prediction) 
    prediction = np.log10(prediction+10e-5)
    true = np.array(true) 
    true = np.log10(true+10e-5)
    true_mean = true.mean()
    
    n = prediction.shape[0] 
    num_2 = (1/n)*(((true - prediction) ** 2).sum()) 
    num_4 = (1/n)*(((true - prediction) ** 4).sum())
    den_2 = (1/n)*(((true - true_mean ) ** 2).sum())
    den_4 = (1/n)*(((true - true_mean ) ** 4).sum())

    r2 = 1 - (num_2/den_2) 
    r2_min = (1 - (1/n)*(((true - prediction) ** 2)/((true - true_mean ) ** 2))).min()
    r2_max = (1 - (1/n)*(((true - prediction) ** 2)/((true - true_mean ) ** 2))).max()
    if score :
        #r2 should be higher than possible so 
        score = (abs(r2)-r2_min/(r2_max-r2_min))
        return r2, score

    return r2, num_2, num_4, den_2, den_4 
    

from scipy import stats

def significatif_test(n, num_2,num_4, den_2, den_4): 
    square_var_den = (den_4 - den_2**2)**0.5/(n)**0.5
    square_var_num = (num_4 - num_2**2)**0.5/(n)**0.5

    return square_var_num, square_var_den

def function_slope(prediction, true, score = False) :
    prediction = np.array(prediction+10e-5) 
    prediction = np.log10(prediction)
    true = np.array(true+10e-5) 
    true = np.log10(true)

    covariance = ((prediction - prediction.mean()) * (true - true.mean())).sum()
    variance   = ((prediction - prediction.mean()) ** 2).sum()

    slope = covariance / variance
    if score : 
        #good score is closer to 1
       one_minus_slope_max = abs(1-(((prediction - prediction.mean()) * (true - true.mean()))/((prediction - prediction.mean()) ** 2))).max()
       one_minus_slope_min = abs(1-(((prediction - prediction.mean()) * (true - true.mean()))/((prediction - prediction.mean()) ** 2))).min()
       score = (abs(1-slope)-one_minus_slope_max)/(one_minus_slope_min-one_minus_slope_max)
       return slope, score
    
    return slope

def function_MAPD(prediction, true, score = False) : 
    y_true, y_pred = np.array(true), np.array(prediction)
    
    median_absolute_error =  np.median(np.abs((y_true - y_pred) / y_true)) 
    median_absolute_percentage_error = median_absolute_error * 100 
    
    if score : 
         median_absolute_error_max =  (abs((y_true - y_pred) / y_true)).max()
         median_absolute_error_min =  (abs((y_true - y_pred) / y_true)).min()
         score = (abs(median_absolute_error) - median_absolute_error_max) /(median_absolute_error_min - median_absolute_error_max)
         return median_absolute_percentage_error, score 
        
    return median_absolute_percentage_error

def function_RMSD(prediction, true) : 
    prediction = np.array(prediction) 
    true = np.array(true) 
    
    residuals = true - prediction 
    n = prediction.shape[0] 
    num_2 = (1/n)*(((true - prediction) ** 2).sum())
    rmsd = num_2**0.5

    return rmsd

import os 
import pandas as pd
import plotly.express as px

def calculate_scores(df):
    # Calculating min and max for each score category
    min_slope = df['SLOPE'].min()
    max_slope = df['SLOPE'].max()
    min_r2 = df['R2'].min()
    max_r2 = df['R2'].max()
    min_mapd = df['MAPD'].min()
    max_mapd = df['MAPD'].max()
    min_rmse = df['RMSD'].min()
    max_rmse = df['RMSD'].max( )

    # Calculating scores
    df.loc[:, 'SLOPE Score'] = (1 - df['SLOPE'] - max(1 - df['SLOPE']) ) / (min(1 - df['SLOPE']) - max(1 - df['SLOPE']))
    df.loc[:, 'R2 Score'] = (df['R2'] - min_r2) / (max_r2 - min_r2)
    df.loc[:, 'MAPD Score'] = (df['MAPD'] - max_mapd) / (min_mapd - max_mapd)
    df.loc[:, 'RMSD Score'] = (df['RMSD'] - max_rmse) / (min_rmse - max_rmse)
    
    # Calculating Global Score
    df.loc[:, 'Global Score'] = df['SLOPE Score'] + df['R2 Score'] + df['MAPD Score'] + df['RMSD Score']
    return df
    
def plot_score(prediction, true, text, latitude, output_link=False, axis_log=True,
               xlabel='True', ylabel='Prediction', PFTS=False, liste_med=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from matplotlib.ticker import ScalarFormatter

    # Conversion tensors -> numpy
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(latitude, torch.Tensor):
        latitude = latitude.detach().cpu().numpy()
    if isinstance(liste_med, torch.Tensor):
        liste_med = liste_med.detach().cpu().numpy()

    prediction = np.ravel(prediction)
    true = np.ravel(true)
    latitude = np.ravel(latitude)

    # ➤ Remplacer les valeurs ≤ 0 par un seuil minimum pour affichage log
    min_val_log = 1e-4
    prediction = np.where(prediction <= 0, min_val_log, prediction)
    true = np.where(true <= 0, min_val_log, true)

    # Calcul des métriques
    N = len(prediction)
    r2, num_2, num_4, den_2, den_4 = function_r2(prediction, true)
    slope = function_slope(prediction, true)
    MAPD = function_MAPD(prediction, true)
    RMSD = function_RMSD(prediction, true)

    # Affichage
    plt.figure(figsize=(10, 6))

    metrics_text = (f'N = {N}\nR² = {r2:.2f}\n'
                    f'MAPD = {MAPD:.2f} %\nRMSD = {RMSD:.4f} mg.m$^3$')
    plt.text(0.02, 0.72, metrics_text, transform=plt.gca().transAxes,
             fontsize=18, multialignment='left')

    # Ligne d'identité
    line = np.linspace(1e-6, 100, 100)
    plt.plot(line, line, color='lightgrey', linestyle='--', linewidth=1.5)

    label = 'Latitude'
    if hasattr(latitude, "name") and latitude.name == "CHL_less_tchla":
        label = 'OC CCI error (tchla_oc_cci - tchla_hplc)'

    use_color = latitude.shape[0] == prediction.shape[0]

    if use_color:
        scatter = plt.scatter(true, prediction, c=latitude, cmap='coolwarm', s=60)
        scatter.set_clim(-90, 90)
        cbar = plt.colorbar(scatter)
        cbar.set_label(label, fontsize=18)
        cbar.ax.tick_params(labelsize=18)
    else:
        plt.scatter(true, prediction, color='gray', s=60)

    # Option coloration Méditerranée
    if liste_med is not None:
        liste_med = np.ravel(liste_med)
        is_med = liste_med == 1
        is_other = ~is_med

        plt.scatter(true[is_other], prediction[is_other], color='#a6a6a6', s=60, label='Other')
        plt.scatter(true[is_med], prediction[is_med],
                    facecolors='#F59D7E',
                    linewidths=1.5, s=60, label='Mediterranean Sea')

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    if axis_log:
        plt.xscale('log')
        plt.yscale('log')

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.yaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.get_major_formatter().set_scientific(False)
    ax.yaxis.get_major_formatter().set_scientific(False)

    # Ticks log personnalisés
    if PFTS:
        ticks = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
        tick_labels = ['0.0001', '0.001', '0.01', '0.1', '1', '10']
    else:
        ticks = [1e-3, 1e-2, 1e-1, 1, 10]
        tick_labels = ['0.001', '0.01', '0.1', '1', '10']

    plt.xlim(min_val_log, 15 + 1.5)
    plt.ylim(min_val_log, 15 + 1.5)

    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, fontsize=20)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=20)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'{text}', fontsize=20)

    if output_link:
        plt.savefig(output_link, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close()



def plot_score_old(prediction, true, text, latitude, output_link=False, axis_log=True,
               xlabel='True', ylabel='Prediction', PFTS=False, liste_med=None):

    import matplotlib.pyplot as plt
    import numpy as np
    import torch

    # Conversion des tenseurs en tableaux NumPy si nécessaire
    if isinstance(prediction, torch.Tensor):
        prediction = prediction.detach().cpu().numpy()
    if isinstance(true, torch.Tensor):
        true = true.detach().cpu().numpy()
    if isinstance(latitude, torch.Tensor):
        latitude = latitude.detach().cpu().numpy()
    if isinstance(liste_med, torch.Tensor):
        liste_med = liste_med.detach().cpu().numpy()

    prediction = np.ravel(prediction)
    true = np.ravel(true)
    latitude = np.ravel(latitude)

    N = len(prediction)
    r2, num_2, num_4, den_2, den_4 = function_r2(prediction, true)
    slope = function_slope(prediction, true)
    MAPD = function_MAPD(prediction, true)
    RMSD = function_RMSD(prediction, true)

    plt.figure(figsize=(10, 6))  # plus grande figure

    metrics_text = (f'N = {N}\nR² = {r2:.2f}\n'
                    f'MAPD = {MAPD:.2f} %\nRMSD = {RMSD:.4f} mg.m$^3$')
    plt.text(0.02, 0.77, metrics_text, transform=plt.gca().transAxes,
             fontsize=16, multialignment='left')

    point = np.arange(-20, 100, 1)
    plt.plot(point, point, color='black', linestyle='-', linewidth=1.5)

    if hasattr(latitude, "name") and latitude.name == "CHL_less_tchla":
        label = 'OC CCI error (tchla_oc_cci - tchla_hplc)'
    else:
        label = 'Latitude'
    
    #scatter = plt.scatter(true, prediction, c=latitude, cmap='coolwarm', s=60)
    scatter = plt.scatter(true, prediction, cmap='coolwarm', s=60)
    #scatter.set_clim(-90, 90)
    #cbar = plt.colorbar(scatter)
    #cbar.set_label(label, fontsize=16)
    #cbar.ax.tick_params(labelsize=16)
    
    if liste_med is not None:
        liste_med = np.ravel(liste_med)
        is_med = liste_med == 1
        is_other = ~is_med

        # Points autres (gris)
        plt.scatter(true[is_other], prediction[is_other], color='#a6a6a6', s=60, label='Other')

        # Points méditerranéens (#F59D7E avec contour noir)
        plt.scatter(true[is_med], prediction[is_med],
                    facecolors='#F59D7E',
                    linewidths=1.5, s=60, label='Mediterranean Sea')

    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)

    if axis_log:
        plt.xscale('log')
        plt.yscale('log')

    plt.gca().set_aspect('equal', adjustable='box')

    max_val = max(max(true), max(prediction))
    margin = 1.5

    if PFTS:
        plt.xlim(10e-5, max_val + margin)
        plt.ylim(10e-5, max_val + margin)
    else:
        plt.xlim(10e-4, max_val + margin)
        plt.ylim(10e-4, max_val + margin)

    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.title(f'Test set - {text} ', fontsize=18)

    if output_link:
        plt.savefig(output_link, bbox_inches='tight')
    plt.tight_layout()
    plt.show()
    plt.close()

def plot_loss(LOSS_TRAIN, LOSS_VAL, title, save_path=None, metric="RMSE"):


    # Style based on metric
    if metric.upper() == "MAPD":
        color_train = '#ff914d'
        color_val = '#ffbd59'
        ylabel = 'MAPD (%)'
        full_title = f'MAPD - {title}'
    else:  # RMSE (default)
        color_train = '#86b499'
        color_val = '#B6EBCC'
        ylabel = 'RMSE'
        full_title = f'Loss function (RMSE) - {title}'

    epoch_train = range(1, len(LOSS_TRAIN) + 1)
    epoch_val = range(1, len(LOSS_TRAIN) + 1, 2)

    plt.plot(epoch_train, LOSS_TRAIN, marker='o', linestyle='-', color=color_train, label='Train')
    plt.plot(epoch_val, LOSS_VAL, marker='o', linestyle='-', color=color_val, label='Validation')

    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()
    plt.title(full_title)

    if save_path:
        base, _ = os.path.splitext(save_path)
        png_path = base + ".png"
        pdf_path = base + ".pdf"
        plt.savefig(png_path, bbox_inches='tight')
        plt.savefig(pdf_path, bbox_inches='tight')
        print(f"Plot saved as:\n  • {png_path}\n  • {pdf_path}")

    plt.show()
    plt.close()