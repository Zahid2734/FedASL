import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
from Code.algorithm.FedAVG_plus_FedPA_algo_mnist import FedAVG_plus_FedPA_algo_mnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 3
num_client= 100
client_percent= .3

file_name = "../../Data/mnist/Dataset0.3_1_100_noise_mnist.pkl"
Dataset= open_file(file_name)

#process and batch the training data for each client
clients= Dataset[0]
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

#process and batch the test set
bad_client= Dataset[1]
x_test= Dataset[2]
y_test= Dataset[3]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2 = FedAVG_plus_FedPA_algo_mnist(num_epochs, num_client, clients_batched,x_test,y_test, client_percent)

sample_list=[GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2]
save_file_name= f'../../Data/mnist/noisy/FedAVG_plus_FedPA_30_percent_1.5_alpha_2_beta.pkl'

save_file(save_file_name,sample_list)