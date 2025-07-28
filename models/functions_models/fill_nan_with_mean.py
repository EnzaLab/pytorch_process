import torch 

def fill_nan_with_mean(tensor, device='cpu'):
    tensor = tensor.to(device)
    
    nan_mask = torch.isnan(tensor)
    mean_values = torch.nanmean(tensor, dim=(2, 3), keepdim=True)
    tensor[nan_mask] = mean_values.expand_as(tensor)[nan_mask]
    
    return tensor

def fill_nan_with_value(tensor, value=9999.0, device='cpu'):
    tensor = tensor.to(device)
    
    nan_mask = torch.isnan(tensor)
    tensor[nan_mask] = value
    
    return tensor