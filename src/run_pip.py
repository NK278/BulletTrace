from loader.dataloader import load_data
# from model_train.RF import train_rf_models
from model_train.mlp_reg import train_mlp
from model_train.cnn_reg import grid_search_cnn
from model_train.GAN_TR import train_cgan
from model_train.new_rf import train_rf_models
# from model_train.new_rf_copy  import train_rf_models

if __name__ == '__main__':
    data_dir='/Users/nishchal_mac/Desktop/FDTD_Bio/Chest_Data'
    # param_grid = {
    #     'filters1': [32, 64],
    #     'filters2': [64, 128],
    #     'lr':       [1e-3, 1e-4],
    #     'batch_size': [16, 32]
    # }
    load_data(data_dir=data_dir)
    # train_rf_models(data_dir=data_dir)
    # train_mlp(train_dir=data_dir)
    # best=grid_search_cnn(data_dir,param_grid=param_grid)
    # train_cgan(data_dir,epochs=100,batch_size=16)
    train_rf_models(data_dir=data_dir,f_low=0,f_high=4e12)
    
    