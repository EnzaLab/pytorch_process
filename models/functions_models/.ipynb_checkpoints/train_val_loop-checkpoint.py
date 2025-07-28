import torch 
import copy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from .usefull_function_model import *

def train_val_loop_test(device, nb_epoch, model_dl, dataloader_train, dataloader_val,
                   optimizer, criterion, loss_diff_threshold, model, pfts=False, patience=50):

    model_dl.to(device)

    LOSS_TRAIN = []
    LOSS_VAL = []
    MAPD_TRAIN = []
    MAPD_VAL = []
    list_val = []

    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model_dl.state_dict())

    epoch_iterator = tqdm(range(nb_epoch), desc="Training epochs")

    for epoch in epoch_iterator:
        total_loss_train = 0
        y_train_true = []
        y_train_pred = []

        model_dl.train()
        for X_batch, Y_batch, __ in dataloader_train:
            if model != 3:
                X_batch[X_batch == -10] = float('nan')
                X_batch = X_batch.nanmean(dim=(2, 3))

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            prediction = model_dl(X_batch.float())

            y_train_pred.extend(prediction.detach().cpu().numpy().flatten())
            y_train_true.extend(Y_batch.cpu().numpy().flatten())

            loss = criterion(prediction, Y_batch if pfts else Y_batch.view(-1, 1))
            loss.backward()
            total_loss_train += loss.item()
            optimizer.step()

        average_loss_train = total_loss_train / len(dataloader_train)
        LOSS_TRAIN.append(average_loss_train)
        MAPD_TRAIN.append(function_MAPD(y_train_pred, y_train_true))

  
        if (epoch + 1) % 2 == 0:
            total_loss_val = 0
            y_val_true = []
            y_val_pred = []

            model_dl.eval()
            with torch.no_grad():
                for X_batch_val, Y_batch_val, __ in dataloader_val:
                    if model != 3:
                        X_batch_val[X_batch_val == -10] = float('nan')
                        X_batch_val = X_batch_val.nanmean(dim=(2, 3))

                    X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                    prediction_val = model_dl(X_batch_val.float())

                    y_val_pred.extend(prediction_val.cpu().numpy().flatten())
                    y_val_true.extend(Y_batch_val.cpu().numpy().flatten())

                    loss_val = criterion(prediction_val, Y_batch_val if pfts else Y_batch_val.view(-1, 1))
                    total_loss_val += loss_val.item()

            average_loss_val = total_loss_val / len(dataloader_val)
            current_mapd_val = function_MAPD(y_val_pred, y_val_true)

            LOSS_VAL.append(average_loss_val)
            MAPD_VAL.append(current_mapd_val)
            list_val.append(current_mapd_val)
            
            if average_loss_val < best_val_loss:
               best_val_loss = average_loss_val
               best_model_state = copy.deepcopy(model_dl.state_dict())
    
        if len(LOSS_TRAIN) >= patience and (epoch + 1) % 2 == 0:
           recent_losses = LOSS_VAL[-patience:]
           if max(recent_losses) - min(recent_losses) < loss_diff_threshold:
                    print(f'Early stopping at epoch {epoch + 1} due to minimal change in validation loss over the last {patience} epochs')
                    break
    
        model_dl.load_state_dict(best_model_state)


    return LOSS_TRAIN, LOSS_VAL, list_val, model_dl, MAPD_TRAIN, MAPD_VAL
def train_val_loop_on_best_MAPD(device, nb_epoch, model_dl, dataloader_train, dataloader_val,
                   optimizer, criterion, loss_diff_threshold, model, pfts=False, patience=50):

    model_dl.to(device)

    LOSS_TRAIN = []
    LOSS_VAL = []
    MAPD_TRAIN = []
    MAPD_VAL = []
    list_val = []

    best_val_mapd = float('inf')
    best_model_state = copy.deepcopy(model_dl.state_dict())

    epoch_iterator = tqdm(range(nb_epoch), desc="Training epochs")

    for epoch in epoch_iterator:
        total_loss_train = 0
        y_train_true = []
        y_train_pred = []

        model_dl.train()
        for X_batch, Y_batch, __ in dataloader_train:
            if model != 3:
                X_batch[X_batch == -10] = float('nan')
                X_batch = X_batch.nanmean(dim=(2, 3))

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            prediction = model_dl(X_batch.float())

            y_train_pred.extend(prediction.detach().cpu().numpy().flatten())
            y_train_true.extend(Y_batch.cpu().numpy().flatten())

            loss = criterion(prediction, Y_batch if pfts else Y_batch.view(-1, 1))
            loss.backward()
            total_loss_train += loss.item()
            optimizer.step()

        average_loss_train = total_loss_train / len(dataloader_train)
        LOSS_TRAIN.append(average_loss_train)
        MAPD_TRAIN.append(function_MAPD(y_train_pred, y_train_true))

        if (epoch + 1) % 2 == 0:
            total_loss_val = 0
            y_val_true = []
            y_val_pred = []

            model_dl.eval()
            with torch.no_grad():
                for X_batch_val, Y_batch_val, __ in dataloader_val:
                    if model != 3:
                        X_batch_val[X_batch_val == -10] = float('nan')
                        X_batch_val = X_batch_val.nanmean(dim=(2, 3))

                    X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                    prediction_val = model_dl(X_batch_val.float())

                    y_val_pred.extend(prediction_val.cpu().numpy().flatten())
                    y_val_true.extend(Y_batch_val.cpu().numpy().flatten())

                    loss_val = criterion(prediction_val, Y_batch_val if pfts else Y_batch_val.view(-1, 1))
                    total_loss_val += loss_val.item()

            average_loss_val = total_loss_val / len(dataloader_val)
            current_mapd_val = function_MAPD(y_val_pred, y_val_true)

            LOSS_VAL.append(average_loss_val)
            MAPD_VAL.append(current_mapd_val)
            list_val.append(current_mapd_val)

            # Meilleur modèle basé sur MAPD
            if current_mapd_val < best_val_mapd:
                best_val_mapd = current_mapd_val
                best_model_state = copy.deepcopy(model_dl.state_dict())

            if len(MAPD_VAL) >= patience:
                recent_mapds = MAPD_VAL[-patience:]
                if max(recent_mapds) - min(recent_mapds) < loss_diff_threshold:
                    print(f'Early stopping at epoch {epoch + 1} due to minimal change in MAPD over the last {patience} epochs')
                    break

    model_dl.load_state_dict(best_model_state)

    return LOSS_TRAIN, LOSS_VAL, list_val, model_dl, MAPD_TRAIN, MAPD_VAL

def train_val_loop(device, nb_epoch, model_dl, dataloader_train, dataloader_val,
                   optimizer, criterion, loss_diff_threshold, model, pfts=False, patience=50,
                   use_penalty=False, penalty_type="L2", penalty_lambda=1e-4):

    model_dl.to(device)
    
    LOSS_TRAIN = []
    LOSS_VAL = []
    list_val = []

    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model_dl.state_dict())

    epoch_iterator = tqdm(range(nb_epoch), desc="Training epochs")

    for epoch in epoch_iterator:
        total_loss_train = 0
        model_dl.train()
        for X_batch, Y_batch, __ in dataloader_train:
            if model != 3:
                X_batch[X_batch == -10] = float('nan')
                X_batch = X_batch.nanmean(dim=(2, 3))

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()
            prediction = model_dl(X_batch.float())

            loss = criterion(prediction, Y_batch if pfts else Y_batch.view(-1, 1))
            loss.backward()
            total_loss_train += loss.item()
            optimizer.step()

        average_loss_train = total_loss_train / len(dataloader_train)
        LOSS_TRAIN.append(average_loss_train)

        if (epoch + 1) % 2 == 0:
            total_loss_val = 0
            model_dl.eval()
            with torch.no_grad():
                for X_batch_val, Y_batch_val, __ in dataloader_val:
                    if model != 3:
                        X_batch_val[X_batch_val == -10] = float('nan')
                        X_batch_val = X_batch_val.nanmean(dim=(2, 3))

                    X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                    prediction_val = model_dl(X_batch_val.float())

                    loss_val = criterion(prediction_val, Y_batch_val if pfts else Y_batch_val.view(-1, 1))
                    total_loss_val += loss_val.item()

            average_loss_val = total_loss_val / len(dataloader_val)
            LOSS_VAL.append(average_loss_val)
            list_val.append(average_loss_val)

            if average_loss_val < best_val_loss:
                best_val_loss = average_loss_val
                best_model_state = copy.deepcopy(model_dl.state_dict())

        if len(LOSS_TRAIN) >= patience and (epoch + 1) % 2 == 0:
            recent_losses = LOSS_VAL[-patience:]
            if max(recent_losses) - min(recent_losses) < loss_diff_threshold:
                print(f'Early stopping at epoch {epoch + 1} due to minimal change in validation loss over the last {patience} epochs')
                break

    model_dl.load_state_dict(best_model_state)
    MAPD_TRAIN, MAPD_VAL = [], []

    return LOSS_TRAIN, LOSS_VAL, list_val, model_dl, MAPD_TRAIN, MAPD_VAL

def train_val_loop_l2(device, nb_epoch, model_dl, dataloader_train, dataloader_val,
                   optimizer, criterion, loss_diff_threshold, model, pfts=False, patience=50,
                   use_penalty=False, penalty_type="L2", penalty_lambda=7*1e-4):

    model_dl.to(device)

    LOSS_TRAIN = []
    LOSS_VAL = []
    list_val = []
    best_val_loss = float('inf')
    best_model_state = copy.deepcopy(model_dl.state_dict())
    epoch_iterator = tqdm(range(nb_epoch), desc="Training epochs")

    for epoch in epoch_iterator:
        total_loss_train = 0
        model_dl.train()

        for X_batch, Y_batch, __ in dataloader_train:
            if model != 3:
                X_batch[X_batch == -10] = float('nan')
                X_batch = X_batch.nanmean(dim=(2, 3))

            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            optimizer.zero_grad()

            prediction = model_dl(X_batch.float())
            
            #tchla_pred = torch.sum(prediction, dim=1, keepdim = True)
            #prediction = torch.cat((prediction, tchla_pred), 1)

            #Y_batch_tchla = torch.sum(Y_batch, dim = 1, keepdim = True)
            #Y_batch = torch.cat((Y_batch, Y_batch_tchla),1)
            loss = criterion(prediction, Y_batch if pfts else Y_batch.view(-1, 1))

            # Add L2 penalty
            if use_penalty and penalty_type == "L2":
                l2_penalty = 0
                for param in model_dl.parameters():
                    l2_penalty += torch.sum(param**2)
                loss += penalty_lambda * l2_penalty

            loss.backward()
            total_loss_train += loss.item()
            optimizer.step()

        average_loss_train = total_loss_train / len(dataloader_train)
        LOSS_TRAIN.append(average_loss_train)

        if (epoch + 1) % 2 == 0:
            total_loss_val = 0
            model_dl.eval()
            with torch.no_grad():
                for X_batch_val, Y_batch_val, __ in dataloader_val:
                    if model != 3:
                        X_batch_val[X_batch_val == -10] = float('nan')
                        X_batch_val = X_batch_val.nanmean(dim=(2, 3))

                    X_batch_val, Y_batch_val = X_batch_val.to(device), Y_batch_val.to(device)
                    prediction_val = model_dl(X_batch_val.float())

                    #tchla_pred_val = torch.sum(prediction_val, dim = 1, keepdim = True)
                    #prediction_val = torch.cat((prediction_val, tchla_pred_val), 1)

                    #Y_batch_val_tchla = torch.sum(Y_batch_val, dim = 1, keepdim = True)
                    #Y_batch_val = torch.cat((Y_batch_val, Y_batch_val_tchla),1)
                    
                    loss_val = criterion(prediction_val, Y_batch_val if pfts else Y_batch_val.view(-1, 1))
                    total_loss_val += loss_val.item()

            average_loss_val = total_loss_val / len(dataloader_val)
            LOSS_VAL.append(average_loss_val)
            list_val.append(average_loss_val)

            if average_loss_val < best_val_loss:
                best_val_loss = average_loss_val
                best_model_state = copy.deepcopy(model_dl.state_dict())

        if len(LOSS_TRAIN) >= patience and (epoch + 1) % 2 == 0:
            recent_losses = LOSS_VAL[-patience:]
            if max(recent_losses) - min(recent_losses) < loss_diff_threshold:
                print(f'Early stopping at epoch {epoch + 1} due to minimal change in validation loss over the last {patience} epochs')
                break

    model_dl.load_state_dict(best_model_state)
    MAPD_TRAIN, MAPD_VAL = [], []

    return LOSS_TRAIN, LOSS_VAL, list_val, model_dl, MAPD_TRAIN, MAPD_VAL
    

