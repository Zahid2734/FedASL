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
plt.rcParams.update({'font.size': 130,'font.weight':'bold','pdf.fonttype':42})

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

# fedavg_com= [1,1,1]
men_means=[round((f1_cifar10_avg+f2_cifar10_avg)/5400,2)*100,round((f1_mnist_avg+f2_mnist_avg)/5400,2)*100, round((f1_femnist_avg+f2_femnist_avg)/75000,2)*100]
women_means=[round((f1_cifar10_tm+f2_cifar10_tm)/5400,2)*100,round((f1_mnist_tm+f2_mnist_tm)/5400,2)*100, round((f1_femnist_tm+f2_femnist_tm)/75000,2)*100]
labels = ['CIFAR10', 'MNIST', 'FEMNIST']

fig,ax = plt.subplots(figsize =(50, 30))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)

x = np.arange(len(labels))  # the label locations
width = 0.33  # the width of the bars

# fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, men_means, width,zorder=3,color='#005CAB',edgecolor ='black', label='FedAVG+FedSRC')
rects2 = ax.bar(x + width/2, women_means, width, zorder=3,color='#E31B23',edgecolor ='black', label='TM+FedSRC')

# Add some text for labels, title and custom x-axis tick labels, etc.
plt.ylabel('Communication \n Saving %', fontweight ='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)
plt.yticks(np.arange(0, 60, 15))
plt.ylim(0, 60)
plt.legend(loc='upper left', ncol=1, fontsize=130, frameon=False, borderaxespad=0.02, handlelength=0.7,
               handletextpad=0.2, labelspacing=0.2, columnspacing=0.6, bbox_to_anchor=(0.008, 1.01))
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)


def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()

        ax.annotate('{}%'.format(int(height)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 3 points vertical offset
                    textcoords="offset points", fontsize=130,color ='black',
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout()

plt.show()
plt.savefig('../../Figures/sigmetrix/communication_save_poster.jpg', bbox_inches='tight')