import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/target/target_shuffling/FedSTD_loss_0.3_1_0.3_100_200_5_target_shuffle_cifar10.pkl '
beta1= open_file(data1)

data2= '../../Data/cifar10/target/target_shuffling/FedSTD_loss_0.3_1_0.3_100_200_10_target_shuffle_cifar10.pkl '
beta2= open_file(data2)

data3= '../../Data/cifar10/target/target_shuffling/FedSTD_loss_0.3_1_0.3_100_200_15_target_shuffle_cifar10.pkl '
beta3= open_file(data3)

data4='../../Data/cifar10/target/target_shuffling/FedSTD_loss_0.3_1_0.3_100_200_20_target_shuffle_cifar10.pkl '
beta4= open_file(data4)

data5= '../../Data/cifar10/target/target_shuffling/FedSTD_loss_0.3_1_0.3_100_200_25_target_shuffle_cifar10.pkl '
beta5= open_file(data5)

data6= '../../Data/cifar10/target/target_shuffling/FedSTD_loss_0.3_1_0.3_100_200_30_target_shuffle_cifar10.pkl '
beta6= open_file(data6)


beta_alpha= [st.mean(beta1[0][150:200])*100,st.mean(beta2[0][150:200])*100,st.mean(beta3[0][150:200])*100,st.mean(beta4[0][150:200])*100,st.mean(beta5[0][150:200])*100,
             st.mean(beta6[0][150:200])*100]

plt.figure(figsize=(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)


plt.plot(beta_alpha, '-o', c='darkgreen', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=25, mfc='yellow', mec='black', mew=1)



my_xticks = ['5','10','15','20','25','30']
x = np.array([0,1,2,3,4,5])
plt.xticks(x, my_xticks)
plt.xlabel("Coordinating Attacker (%)", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(30, 91, 15))
plt.ylim(30, 90)

plt.show
plt.savefig('../../Figures/edge/power point/target_shuffle.png', bbox_inches='tight')