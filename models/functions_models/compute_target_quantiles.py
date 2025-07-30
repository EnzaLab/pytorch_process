import torch 

def compute_target_quantiles(train_loader, quantile=0.15, pfts=False, device='cpu'):
    all_targets = []

    for targets in train_loader:
        targets = targets[1].to(device)
        all_targets.append(targets)

    all_targets = torch.cat(all_targets, dim=0)

    if not pfts:
        all_targets = all_targets.view(-1)
        q_low = torch.quantile(all_targets, quantile)
        q_high = torch.quantile(all_targets, 1 - quantile)
        return q_low, q_high
    else:
        q_low = []
        q_high = []
        for i in range(all_targets.shape[1]):
            col = all_targets[:, i]
            q_low.append(torch.quantile(col, quantile))
            q_high.append(torch.quantile(col, 1 - quantile))
        return torch.stack(q_low), torch.stack(q_high)
    