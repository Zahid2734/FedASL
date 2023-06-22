import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
from Code.utils.file_handler import save_file, open_file


def bad_data_counter(filter1,filter2, starting_point):
    sum_filter1=0
    sum_filter2=0
    for i in range(starting_point, len(filter1)):
        sum_filter1= sum_filter1+len(filter1[i])
        bla= list(set(filter2[i])-set(filter1[i]))
        sum_filter2 =sum_filter2+len(bla)

    return sum_filter1, sum_filter2

plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/noisy/FedAVG_0.3_1_0.3_100_200_noise_cifar10.pkl'
FedAVG_cifar10= open_file(data1)

data2= '../../Data/cifar10/noisy/FedAVG_plus_FedPA_0.3_1_0.3_100_200_noise_cifar10_11.pkl'
FedAVG_FedPA_cifar10= open_file(data2)
f1_cifar10_avg,f2_cifar10_avg= bad_data_counter(FedAVG_FedPA_cifar10[5],FedAVG_FedPA_cifar10[6],10)
print("FedAvg_cifar10", f1_cifar10_avg,f2_cifar10_avg)

data3= '../../Data/cifar10/noisy/trimmed_mean_plus_FedPA_0.3_1_0.3_100_200_noise_cifar10_11.pkl'
TM_FedPA_cifar10= open_file(data3)
f1_cifar10_tm,f2_cifar10_tm= bad_data_counter(TM_FedPA_cifar10[5],TM_FedPA_cifar10[6],10)
print("FedTm_cifar10", f1_cifar10_tm,f2_cifar10_tm)

data4= '../../Data/cifar10/noisy/trimmed_mean_0.3_1_0.3_100_200_noise_cifar10.pkl'
TM_cifar10= open_file(data4)


data5= '../../Data/mnist/noisy/FedAVG_0.3_1_0.3_100_200_noise_mnist.pkl'
FedAVG_mnist= open_file(data5)

data6= '../../Data/mnist/noisy/FedAVG_plus_FedPA_0.3_1_0.3_100_200_noise_mnist_11.pkl'
FedAVG_FedPA_mnist= open_file(data6)
f1_mnist_avg,f2_mnist_avg= bad_data_counter(FedAVG_FedPA_mnist[5],FedAVG_FedPA_mnist[6],10)
print("FedAvg_mnist", f1_mnist_avg,f2_mnist_avg)

data7= '../../Data/mnist/noisy/trimmed_mean_plus_FedPA_0.3_1_0.3_100_200_noise_mnist_11.pkl'
TM_FedPA_mnist= open_file(data7)
f1_mnist_tm,f2_mnist_tm= bad_data_counter(TM_FedPA_mnist[5],TM_FedPA_mnist[6],10)
print("Fedtm_mnist", f1_mnist_tm,f2_mnist_tm)

data8= '../../Data/mnist/noisy/trimmed_mean_0.3_1_0.3_100_200_noise_mnist.pkl'
TM_mnist= open_file(data8)


data9= '../../Data/Femnist/Noisy/FedAVG_0.3_1_3000_250_noise_Femnist.pkl'
FedAVG_femnit= open_file(data9)

data10= '../../Data/Femnist/Noisy/FedAVG_plus_FedPA_0.3_1_3000_250_noise_Femnist_11.pkl'
FedAVG_FedPA_femnist= open_file(data10)
f1_femnist_avg,f2_femnist_avg= bad_data_counter(FedAVG_FedPA_femnist[5],FedAVG_FedPA_femnist[6],10)
print("FedAvg_femnist", f1_femnist_avg,f2_femnist_avg)

data11= '../../Data/Femnist/Noisy/trimmed_mean_plus_FedPA_0.3_1_3000_250_noise_Femnist_11.pkl'
TM_FedPA_femnist= open_file(data11)
f1_femnist_tm,f2_femnist_tm= bad_data_counter(TM_FedPA_femnist[5],TM_FedPA_femnist[6],10)
print("Fedpa_femnist", f1_femnist_tm,f2_femnist_tm)


data12= '../../Data/Femnist/Noisy/trimmed_mean_0.3_1_3000_250_noise_Femnist.pkl'
TM_femnist= open_file(data12)

fedavg_com= [1,1,1]
fedavg_fedpa_com=[1-(f1_cifar10_avg+f2_cifar10_avg)/5100,1-(f1_mnist_avg+f2_mnist_avg)/5100, 1-(f1_femnist_avg+f2_femnist_avg)/60000]
tm_fedpa_com=[1-(f1_cifar10_tm+f2_cifar10_tm)/5100,1-(f1_mnist_tm+f2_mnist_tm)/5100, 1-(f1_femnist_tm+f2_femnist_tm)/60000]

# set width of bar
barWidth = .25
fig,ax = plt.subplots(figsize =(10, 6))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)

# Set position of bar on X axis
br1 = np.arange(len(fedavg_com))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, fedavg_com,zorder=3, color='#005CAB', width = barWidth,edgecolor ='black', label ='FedAVG/ TM')
plt.bar(br2, fedavg_fedpa_com,zorder=3, color='#ffbf00', width = barWidth,edgecolor ='black', label ='FedAVG+FedSRC')
plt.bar(br3, tm_fedpa_com,zorder=3, color='#E31B23', width = barWidth,edgecolor ='black', label ='TM+FedSRC')

# Adding Xticks
ax.set_xticks(br1+barWidth)
ax.set_xticklabels( ('CIFAR10', 'MNIST', 'FEMNIST') )
plt.ylabel(' Normalized \n Comnunication', fontweight ='bold')

plt.yticks(np.arange(0, 1.3, .25))
plt.ylim(0, 1.3)
plt.legend(loc='upper left', ncol=2, fontsize=32, frameon=False, borderaxespad=0.02, handlelength=0.7,
               handletextpad=0.2, labelspacing=0.2, columnspacing=0.6, bbox_to_anchor=(0.008, 1.01))
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
plt.show()
plt.savefig('../../Figures/sigmetrix/sigmetriz_communication_save.pdf', bbox_inches='tight')

