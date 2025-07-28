import torch

def current_norm_angle(U, V):
    norm = torch.sqrt(U**2 + V**2)
    
    cos_theta = U / norm
    sin_theta = V / norm
    
    return norm, cos_theta, sin_theta

def add_current(tensor): 
    U = tensor[:,2,:,:]
    V = tensor[:,3,:,:]

    norm, cos_theta, sin_theta = current_norm_angle(U, V)

    tensor[:,2,:,:] = cos_theta
    tensor[:,3,:,:] = sin_theta
    norm_expanded = norm.unsqueeze(1)
    part1 = tensor[:, :4, :, :]
    part2 = tensor[:, 4:, :, :]
    tensor = torch.cat((part1, norm_expanded, part2), dim=1)

    return tensor