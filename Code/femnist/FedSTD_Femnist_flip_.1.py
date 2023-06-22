import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data_femnist, creating_shuffling_clients,create_client
from Code.algorithm.FedSTD_algo_Femnist import FedSTD_algo_Femnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 250
num_client= 3000
client_percent= .1
alpha=1
beta= .2

file_name = "../../Data/Femnist/Dataset_flip_3000_0.1_1.pkl"
Dataset= open_file(file_name)

file_name = "../../Data/Femnist/Testset_3000.pkl"
Testset= open_file(file_name)


#process and batch the training data for each client
clients= Dataset[0]
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data_femnist(data)

#process and batch the test set
bad_client= Dataset[1]
x_test= Testset[0]
y_test= Testset[1]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

previous_client=1
GLobal_accuracy_STD,Client_accuracy_STD,  scaled_weight_STD, taken_client = FedSTD_algo_Femnist(num_epochs, num_client, clients_batched,x_test,y_test, client_percent,alpha,beta,previous_client)


sample_list=[GLobal_accuracy_STD,Client_accuracy_STD,scaled_weight_STD, bad_client, taken_client]

save_file_name= f'../../Data/Femnist/flipping/FedSTD_0.1_1_{num_client}_{num_epochs}_{alpha}_{beta}_flip_Femnist.pkl'
save_file(save_file_name,sample_list)