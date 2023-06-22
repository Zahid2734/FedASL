import pickle
import numpy as np
from Code.utils.math_function import median_std
from Code.utils.file_handler import save_file, open_file




data5= '../../Data/Femnist/clean/FedAVG_3000_250_clean_Femnist.pkl'
FedAVG= open_file(data5)

data6= '../../Data/Femnist/clean/FedSTD_3000_250_0.5_0.2_clean_Femnist.pkl'
FedSTD= open_file(data6)

data7= '../../Data/Femnist/clean/FedSTD_loss_3000_250_0.5_0.2_clean_Femnist.pkl'
FedSTD_loss= open_file(data7)

data8= '../../Data/Femnist/clean/median_3000_250_clean_Femnist.pkl'
median= open_file(data8)

data9= '../../Data/Femnist/clean/trimmed_mean_3000_250_clean_Femnist.pkl'
TM= open_file(data9)



med, std, mean =median_std(FedAVG[0][200:250])
print('FedAVG',med,std, mean)

med, std, mean =median_std(FedSTD_loss[0][200:250])
print('FedSTD',med,std, mean)

med, std, mean=median_std(median[0][200:250])
print('Median',med,std, mean)

med, std, mean=median_std(TM[0][200:250])
print('TM',med,std, mean)
