import os

def create_new_training_folder(base_path, pfts=False):
    import re
    
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    
    existing_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]

    train_dirs = [d for d in existing_dirs if re.match(r'^train_\d+$', d)]
    if pfts : 
        train_dirs = [d for d in existing_dirs if re.match(r'^train_pfts_\d+$', d)]

    indices = [int(re.search(r'\d+', d).group()) for d in train_dirs]
    next_index = max(indices) + 1 if indices else 1

    if pfts :
        new_dir_name = f"train_pfts_{next_index}"
    else : 
        new_dir_name = f"train_{next_index}"
        
    new_dir_path = os.path.join(base_path, new_dir_name)
    os.makedirs(new_dir_path, exist_ok=True)

    return new_dir_path