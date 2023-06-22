import pickle
import numpy as np
from Code.utils.math_function import median_std
from Code.utils.file_handler import save_file, open_file



data5= '../../Data/mnist/flipping/table/FedAVG_0.4_1_0.3_100_200_flip_mnist.pkl'
FedAVG= open_file(data5)

data7= '../../Data/mnist/flipping/table/FedSTD_loss_0.4_1_0.3_100_200_0.9_0.2_flip_mnist.pkl'
FedSTD_loss= open_file(data7)

data8= '../../Data/mnist/flipping/table/median_0.4_1_0.3_100_200_flip_mnist.pkl'
median= open_file(data8)

data9= '../../Data/mnist/flipping/table/trimmed_mean_0.4_1_0.3_100_200_flip_mnist.pkl'
TM= open_file(data9)


med, std, mean =median_std(FedAVG[0][150:200])
print('FedAVG',med,std, mean)

med, std, mean =median_std(FedSTD_loss[0][150:200])
print('FedSTD',med,std, mean)

med, std, mean=median_std(median[0][150:200])
print('Median',med,std, mean)

med, std, mean=median_std(TM[0][150:200])
print('TM',med,std, mean)