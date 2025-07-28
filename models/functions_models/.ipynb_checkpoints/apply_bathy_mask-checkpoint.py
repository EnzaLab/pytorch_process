import torch

def apply_bathy_mask(tensor): 

    bathy = tensor[:, 18, :, :]
    mask = bathy >= 0 
    mask_expanded = mask.unsqueeze(1).expand(-1, 18, -1, -1)
    data_masked = tensor[:, :18, :, :].clone()
    data_masked[mask_expanded] = -10
    data_masked_full = torch.cat([data_masked, tensor[:, 18:19, :, :]], dim=1)

    return data_masked_full