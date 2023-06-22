import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
from Code.algorithm.FedAVG_plus_FedPA_algo_cifar10_sig import FedAVG_plus_FedPA_algo_cifar10
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 2
num_client= 100
client_percent= .3
file_name = "../../Data/Femnist/Dataset_noise_3000_0.3_1.pkl"
Dataset= open_file(file_name)
data = Dataset[0]
sla={}
sla['client_1']= batch_data(data['clients_1'][0:100])
client_names = list(data.keys())
sla[client_names[1]]= batch_data(data[client_names[1]][0:100])