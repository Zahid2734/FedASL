import pickle
import numpy as np
from Code.utils.math_function import median_std
from Code.utils.file_handler import save_file, open_file





data5= '../../Data/Femnist/Shuffling/table/FedAVG_0.4_1_3000_250_shuffle_Femnist.pkl'
FedAVG= open_file(data5)

data7= '../../Data/Femnist/Shuffling/table/FedSTD_loss_0.4_1_3000_250_0.7_0.2_shuffle_Femnist.pkl'
FedSTD_loss= open_file(data7)

data8= '../../Data/Femnist/Shuffling/table/median_0.4_1_3000_250_shuffle_Femnist.pkl'
median= open_file(data8)

data9= '../../Data/Femnist/Shuffling/table/trimmed_mean_0.4_1_3000_250_shuffle_Femnist.pkl'
TM= open_file(data9)


med, std, mean =median_std(FedAVG[0][200:250])
print('FedAVG',med,std, mean)

med, std, mean =median_std(FedSTD_loss[0][200:250])
print('FedSTD',med,std, mean)

med, std, mean=median_std(median[0][200:250])
print('Median',med,std, mean)

med, std, mean=median_std(TM[0][200:250])
print('TM',med,std, mean)
