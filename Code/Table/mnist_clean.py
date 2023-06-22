import pickle
import numpy as np
from Code.utils.math_function import median_std
from Code.utils.file_handler import save_file, open_file



data5= '../../Data/mnist/clean/FedAVG_0.3_100_200_clean_mnist.pkl'
FedAVG= open_file(data5)


data7= '../../Data/mnist/clean/FedSTD_loss_0.3_100_200_0.2_0.05_clean_mnist.pkl'
FedSTD_loss= open_file(data7)

data8= '../../Data/mnist/clean/median_0.3_100_200_clean_mnist.pkl'
median= open_file(data8)

data9= '../../Data/mnist/clean/trimmed_mean_0.3_100_200_clean_mnist.pkl'
TM= open_file(data9)


med, std, mean =median_std(FedAVG[0][150:200])
print('FedAVG',med,std, mean)

med, std, mean =median_std(FedSTD_loss[0][150:200])
print('FedSTD',med,std, mean)

med, std, mean=median_std(median[0][150:200])
print('Median',med,std, mean)

med, std, mean=median_std(TM[0][150:200])
print('TM',med,std, mean)

