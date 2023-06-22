import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/data percent/new/FedVG_0.pkl'
beta1= open_file(data1)

data2= '../../Data/cifar10/data percent/new/FedVG_.1.pkl'
beta2= open_file(data2)

data3= '../../Data/cifar10/data percent/new/FedVG_.3.pkl '
beta3= open_file(data3)

data4='../../Data/cifar10/data percent/new/FedVG_.5.pkl'
beta4= open_file(data4)

data5= '../../Data/cifar10/data percent/new/FedVG_.75.pkl'
beta5= open_file(data5)

data6= '../../Data/cifar10/data percent/new/FedVG_1.pkl'
beta6= open_file(data6)

#FedSTD
data7= '../../Data/cifar10/data percent/new/FedSTD_loss_0.pkl'
beta7= open_file(data7)

data8= '../../Data/cifar10/data percent/new/FedSTD_loss_.1.pkl'
beta8= open_file(data8)


data9= '../../Data/cifar10/data percent/new/FedSTD_loss_.3.pkl'
beta9= open_file(data9)

data10= '../../Data/cifar10/data percent/new/FedSTD_loss_.5.pkl'
beta10= open_file(data10)

data11= '../../Data/cifar10/data percent/new/FedSTD_loss_.75.pkl'
beta11= open_file(data11)

data12= '../../Data/cifar10/data percent/new/FedSTD_loss_1.pkl'
beta12= open_file(data12)

#median
data13= '../../Data/cifar10/data percent/new/median__0.pkl'
beta13= open_file(data13)

data14= '../../Data/cifar10/data percent/new/median_.1.pkl'
beta14= open_file(data14)

data15= '../../Data/cifar10/data percent/new/median_.3.pkl '
beta15= open_file(data15)

data16= '../../Data/cifar10/data percent/new/median_.5.pkl'
beta16= open_file(data16)

data17= '../../Data/cifar10/data percent/new/median_.75.pkl'
beta17= open_file(data17)

data18= '../../Data/cifar10/data percent/new/median_1.pkl'
beta18= open_file(data18)

#TM
data19= '../../Data/cifar10/data percent/new/trimmed_mean_0.pkl'
beta19= open_file(data19)

data20= '../../Data/cifar10/data percent/new/trimmed_mean_.1.pkl'
beta20= open_file(data20)

data21= '../../Data/cifar10/data percent/new/trimmed_mean_.3.pkl'
beta21= open_file(data21)

data22= '../../Data/cifar10/data percent/new/trimmed_mean_.5.pkl'
beta22= open_file(data22)

data23= '../../Data/cifar10/data percent/new/trimmed_mean_.75.pkl'
beta23= open_file(data23)

data24= '../../Data/cifar10/data percent/new/trimmed_mean_1.pkl'
beta24= open_file(data24)



beta_alpha= [st.median(beta1[0][125:150])*100,st.median(beta2[0][125:150])*100,st.median(beta3[0][125:150])*100,st.median(beta4[0][125:150])*100,st.median(beta5[0][125:150])*100,
             st.median(beta6[0][125:150])*100,st.median(beta7[0][150:200])*100,st.median(beta8[0][150:200])*100,st.median(beta9[0][150:200])*100,st.median(beta10[0][150:200])*100,
             st.median(beta11[0][150:200])*100,st.median(beta12[0][150:200])*100,st.median(beta13[0][150:200])*100,st.median(beta14[0][150:200])*100,st.median(beta15[0][150:200])*100,
             st.median(beta16[0][150:200])*100,st.median(beta17[0][150:200])*100,st.median(beta18[0][150:200])*100,st.median(beta19[0][150:200])*100,st.median(beta20[0][150:200])*100,
             st.median(beta21[0][150:200])*100,st.median(beta22[0][150:200])*100,st.median(beta23[0][150:200])*100,st.median(beta24[0][150:200])*100]

plt.figure(figsize=(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)


plt.plot(beta_alpha[0:6], '-o', c='red', linewidth=10.0, zorder=3, marker='o',markevery=1,markersize=25, mfc='white', mec='red', mew=1,label='FedAVG')
plt.plot(beta_alpha[6:12], '-o', c='blue', linewidth=10.0, zorder=4, marker='D',markevery=1,markersize=25, mfc='white', mec='blue', mew=1,label='FedASL')
plt.plot(beta_alpha[18:24], '-o', c='darkgreen', linewidth=10.0, zorder=2, marker='^',markevery=1,markersize=25, mfc='white', mec='darkgreen', mew=1,label='Trimmed Mean')
plt.plot(beta_alpha[12:18], '-o', c='magenta', linewidth=10.0, zorder=1, marker='*',markevery=1,markersize=25, mfc='white', mec='magenta', mew=1,label='Median')


plt.legend(loc=4)
my_xticks = ['0','10','30','50','75','100']
x = np.array([0,1,2,3,4,5])
plt.xticks(x, my_xticks)
plt.yticks(np.arange(47, 66, 3))
plt.ylim(47, 65)
plt.xlabel("Bad Data (%)", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.legend(bbox_to_anchor=(0,0,.6,.3), loc="lower left", fontsize=30,
                mode="expand", ncol=1)
plt.show
plt.savefig('../../Figures/edge/power point/data_percent_effect.png', bbox_inches='tight')

