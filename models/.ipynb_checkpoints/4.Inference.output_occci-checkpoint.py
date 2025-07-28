# Make inference output for oc cci 
import os 
import torch 

def main():

    login = '/home/elabourdette'
    space_buffer = 50
    year  = 2002
    month = 5
    for day in range(1,32):
        path  = f"{login}/complex/share/save_training_Enza/inference/{year}/{month}/input/{day}/transf_picture_space_{space_buffer}_{year}{month}{day}.pt"
        df_picture = torch.load(path)
    
        CHL = df_picture[:,14,:,:]
        MICRO = df_picture[:,15,:,:]
        NANO = df_picture[:,16,:,:]
        PICO = df_picture[:,17,:,:]
    
        CHL_mean = torch.nanmean(CHL, dim=(1, 2)).unsqueeze(1)
        MICRO_mean = torch.nanmean(MICRO, dim=(1, 2)).unsqueeze(1)
        NANO_mean = torch.nanmean(NANO, dim=(1, 2)).unsqueeze(1)
        PICO_mean = torch.nanmean(PICO, dim=(1, 2)).unsqueeze(1)
    
        picture = torch.load(f'/home/elabourdette/complex/share/save_training_Enza/inference/{year}/{month}/input/{day}/transf_df_space_{space_buffer}_{year}{month}{day}.pt')
        
        CHL_map = torch.cat((picture, CHL_mean), dim=1)
        MICRO_map = torch.cat((picture, MICRO_mean), dim=1)
        NANO_map = torch.cat((picture, NANO_mean), dim=1)
        PICO_map = torch.cat((picture, PICO_mean), dim=1)
    
        path_to_save = f"{login}/complex/share/save_training_Enza/inference/{year}/{month}/output/OCCCI/{day}/"
        os.makedirs(path_to_save, exist_ok=True)
        torch.save(CHL_map, os.path.join(path_to_save,f'prediction_CHL_space_{space_buffer}_{year}{month}{day}.pt'))
        torch.save(MICRO_map, os.path.join(path_to_save,f'prediction_MICRO_space_{space_buffer}_{year}{month}{day}.pt'))
        torch.save(NANO_map, os.path.join(path_to_save,f'prediction_NANO_space_{space_buffer}_{year}{month}{day}.pt'))
        torch.save(PICO_map, os.path.join(path_to_save,f'prediction_PICO_space_{space_buffer}_{year}{month}{day}.pt'))
    
if __name__ == "__main__":
    main()   