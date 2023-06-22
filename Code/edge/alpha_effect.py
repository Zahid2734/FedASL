import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy import linspace
import statistics as st
from Code.utils.file_handler import save_file, open_file

plt.rcParams["font.family"] = "Palatino linotype"
plt.rcParams.update({'font.size': 40,'font.weight':'bold','pdf.fonttype':42})

data1= '../../Data/cifar10/beta alpha/new/alpha_.01.pkl '
beta1= open_file(data1)

data2= '../../Data/cifar10/beta alpha/new/alpha_0.1.pkl '
beta2= open_file(data2)

data3= '../../Data/cifar10/beta alpha/new/alpha_0.2.pkl '
beta3= open_file(data3)

data4='../../Data/cifar10/beta alpha/new/alpha_0.3.pkl '
beta4= open_file(data4)

data5= '../../Data/cifar10/beta alpha/new/alpha_0.4.pkl '
beta5= open_file(data5)

data6= '../../Data/cifar10/beta alpha/new/alpha_0.5.pkl '
beta6= open_file(data6)

data7= '../../Data/cifar10/beta alpha/new/alpha_0.6.pkl '
beta7= open_file(data7)

data8= '../../Data/cifar10/beta alpha/new/alpha_0.7.pkl '
beta8= open_file(data8)

data9= '../../Data/cifar10/beta alpha/new/alpha_0.8.pkl '
beta9= open_file(data9)

data10= '../../Data/cifar10/beta alpha/new/alpha_0.9.pkl '
beta10= open_file(data10)

data11= '../../Data/cifar10/beta alpha/new/alpha_1.pkl '
beta11= open_file(data11)

data12= '../../Data/cifar10/beta alpha/new/alpha_1.1.pkl '
beta12= open_file(data12)

data13= '../../Data/cifar10/beta alpha/new/alpha_1.2.pkl '
beta13= open_file(data13)

data14= '../../Data/cifar10/beta alpha/new/alpha_1.3.pkl '
beta14= open_file(data14)

data15= '../../Data/cifar10/beta alpha/new/alpha_1.4.pkl '
beta15= open_file(data15)

data16= '../../Data/cifar10/beta alpha/new/alpha_1.5.pkl '
beta16= open_file(data16)

data17= '../../Data/cifar10/beta alpha/new/alpha_1.6.pkl '
beta17= open_file(data17)

data18= '../../Data/cifar10/beta alpha/new/alpha_1.7.pkl '
beta18= open_file(data18)

data19= '../../Data/cifar10/beta alpha/new/alpha_1.8.pkl '
beta19= open_file(data19)

data20= '../../Data/cifar10/beta alpha/new/alpha_1.9.pkl '
beta20= open_file(data20)

data21= '../../Data/cifar10/beta alpha/new/alpha_2.pkl '
beta21= open_file(data21)

data22= '../../Data/cifar10/beta alpha/new/alpha_2.1.pkl '
beta22= open_file(data22)

data23= '../../Data/cifar10/beta alpha/new/alpha_2.2.pkl '
beta23= open_file(data23)

data24= '../../Data/cifar10/beta alpha/new/alpha_2.3.pkl '
beta24= open_file(data24)

data25= '../../Data/cifar10/beta alpha/new/alpha_2.4.pkl '
beta25= open_file(data25)

data26= '../../Data/cifar10/beta alpha/new/alpha_2.5.pkl '
beta26= open_file(data26)

data27= '../../Data/cifar10/beta alpha/new/alpha_2.6.pkl '
beta27= open_file(data27)

data28= '../../Data/cifar10/beta alpha/new/alpha_2.7.pkl '
beta28= open_file(data28)

data29= '../../Data/cifar10/beta alpha/new/alpha_2.8.pkl '
beta29= open_file(data29)

data30= '../../Data/cifar10/beta alpha/new/alpha_2.9.pkl '
beta30= open_file(data30)

data31= '../../Data/cifar10/beta alpha/new/alpha_3.pkl '
beta31= open_file(data31)

beta_alpha= [st.median(beta1[0][150:200])*100,st.median(beta2[0][150:200])*100,st.median(beta3[0][150:200])*100,st.median(beta4[0][150:200])*100,st.median(beta5[0][150:200])*100,
             st.median(beta6[0][150:200])*100,st.median(beta7[0][150:200])*100,st.median(beta8[0][150:200])*100,st.median(beta9[0][150:200])*100,st.median(beta10[0][150:200])*100,
             st.median(beta11[0][150:200])*100,st.median(beta12[0][150:200])*100,st.median(beta13[0][150:200])*100,st.median(beta14[0][150:200])*100,st.median(beta15[0][150:200])*100,
             st.median(beta16[0][150:200])*100,st.median(beta17[0][150:200])*100,st.median(beta18[0][150:200])*100,st.median(beta19[0][150:200])*100,st.median(beta20[0][150:200])*100,
             st.median(beta21[0][150:200])*100,st.median(beta22[0][150:200])*100,st.median(beta23[0][150:200])*100,st.median(beta24[0][150:200])*100,st.median(beta25[0][150:200])*100,
             st.median(beta26[0][150:200])*100,st.median(beta27[0][150:200])*100,st.median(beta28[0][150:200])*100,st.median(beta29[0][150:200])*100,st.median(beta30[0][150:200])*100,st.median(beta31[0][150:200])*100]

plt.figure(figsize=(10, 8))
plt.grid(zorder=1, color='#999999', linewidth=2.0, linestyle='--', alpha=0.5)
fedavg= [53.53 for i in range(31)]

plt.plot(beta_alpha, '-o', c='blue', linewidth=10.0, zorder=5, marker='D',markevery=1,markersize=12, mfc='yellow', mec='black', mew=1,label='FedASL')
plt.plot(fedavg, ':', c='red', linewidth=10.0, zorder=4,label='FedAVG')


plt.legend(loc=4)
my_xticks = ['0.01','0.5','1','1.5','2','2.5','3']
x = np.array([0,5,10,15,20,25,30])
plt.xticks(x, my_xticks)
plt.xlabel(r"$\alpha$ Value ($\beta$=$\alpha$) ", weight='bold')
plt.ylabel('Accuracy (%)', weight='bold')
plt.yticks(np.arange(51, 65, 3.00))
plt.ylim(51, 63)
plt.legend(bbox_to_anchor=(.45,0.7,.5,.3), loc="lower left",fontsize=32,
                mode="expand", ncol=1)
plt.show
plt.savefig('../../Figures/edge/power point/alpha_effect.png', bbox_inches='tight')

