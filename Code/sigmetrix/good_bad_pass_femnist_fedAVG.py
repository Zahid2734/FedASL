from Code.utils.file_handler import save_file, open_file
import matplotlib.pyplot as plt
import numpy as np

file_name = "../../Data/Femnist/Dataset_noise_3000_0.3_1.pkl"
Dataset= open_file(file_name)
data10= '../../Data/Femnist/Noisy/FedAVG_plus_FedPA_0.3_1_3000_250_noise_Femnist_11.pkl'
FedAVG_FedPA_femnist= open_file(data10)
bad_client= Dataset[1]
selected_client= FedAVG_FedPA_femnist[3]
filter1_list= FedAVG_FedPA_femnist[5]
filter2_list= FedAVG_FedPA_femnist[6]

def bad_data_counter(filter1,filter2,selected_client, bad_client, starting_point):
    bad_passed=[]
    good_blocked=[]
    total_bad_client=[]
    total_fliter_block=[]
    bad_blocked=[]
    for i in range(starting_point, len(filter1)):
        bad_client_ID= list(set(selected_client[i]) & set(bad_client))
        total_bad_client.append(len(bad_client_ID))
        total_block_ID= list(set().union(filter1[i],filter2[i]))
        total_fliter_block.append(len(total_block_ID))
        bad_block_ID= list(set(bad_client_ID) & set(total_block_ID))
        bad_blocked.append(len(bad_block_ID))
        bad_passed.append(len(bad_client_ID)-len(bad_block_ID))
        good_blocked.append(len(total_block_ID)-len(bad_block_ID))

    return total_bad_client,total_fliter_block,bad_passed, good_blocked, bad_blocked

total_bad_client,total_fliter_block,bad_passed, good_blocked, bad_blocked = bad_data_counter(filter1_list,filter2_list,selected_client, bad_client, starting_point=51)
bad_client= np.array(bad_passed)
good_client= np.array(good_blocked)


plt.rcParams["font.family"] = "palatino linotype"
plt.rcParams.update({'font.size': 25,'font.weight':'bold','pdf.fonttype':42})
plt.figure(figsize=(7.3,5))
plt.grid(zorder=1, color='#999999', linestyle='--', alpha=0.4)
plt.plot((good_client*100)/2700, '-', c='red', linewidth=4.0, zorder=1, mfc='red', mec='red', mew=1, label='Good Clients')
plt.plot(((300-bad_client)*100)/300, '-', c='green', linewidth=4.0, zorder=2,  mfc='green', mec='green', mew=1, label='Bad Clients')


plt.xticks(np.arange(0.0, 201, 50))
plt.title('FedAVG+FedSRC',weight='bold')
plt.xlabel("Rounds of Training", weight='bold')
plt.ylabel('Clients Blocking(%)', weight='bold')
plt.yticks(np.arange(0, 101, 25))
plt.ylim(0, 110)
plt.legend(bbox_to_anchor=(.4,.2,.6,0.2), loc="lower left",
                mode="expand", ncol=1,handlelength=1,
               handletextpad=0.2, labelspacing=0.2)
plt.tight_layout(pad=0.0, h_pad=0.0, w_pad=0.0)
plt.show
# plt.savefig('../../Figures/sigmetrix/good_bad_pass_femnist_fedavg_poster.png', bbox_inches='tight')
