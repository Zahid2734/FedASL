import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file
#for 0.1%
data3= '../../Data/cifar10/shuffling/FedAVG_0.1_1_0.3_100_200_shuffle_cifar10.pkl'
FedAVG_1= open_file(data3)

data4= '../../Data/cifar10/shuffling/FedSTD_0.1_1_0.3_100_200_1_0.2_shuffle_cifar10.pkl'
FedSTD_1= open_file(data4)
# for 0.3%
data1= '../../Data/cifar10/shuffling/FedAVG_0.3_1_0.3_100_200_shuffle_cifar10.pkl'
FedAVG_3= open_file(data1)

data7= '../../Data/cifar10/shuffling/median_0.3_1_0.3_100_200_shuffle_cifar10.pkl'
med_3= open_file(data7)

data8= '../../Data/cifar10/shuffling/trimmed_mean_0.3_1_0.3_100_200_shuffle_cifar10.pkl'
tm_3= open_file(data8)
data2= '../../Data/cifar10/shuffling/FedSTD_0.3_1_0.3_100_200_0.9_0.2_shuffle_cifar10.pkl'
FedSTD_3= open_file(data2)

#for 0.4%
data5= '../../Data/cifar10/shuffling/FedAVG_0.4_1_0.3_100_200_shuffle_cifar10.pkl'
FedAVG_4= open_file(data5)

data6= '../../Data/cifar10/shuffling/FedSTD_0.4_1_0.3_100_200_0.7_0.2_shuffle_cifar10.pkl'
FedSTD_4= open_file(data6)

