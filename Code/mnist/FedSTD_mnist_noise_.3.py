# implementing FL
import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data
from Code.algorithm.FedSTD_algo import FedSTD_algo
from Code.utils.file_handler import save_file, open_file



# user will define
num_epochs= 200
num_client= 100
alpha= .9
beta= .2
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


previous_average= 1
GLobal_accuracy_STD,Client_accuracy_STD,scaled_weight_STD, taken_client= FedSTD_algo(num_epochs, num_client, clients_batched,x_test,y_test,client_percent,alpha,beta,previous_average)
sample_list=[GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, bad_client, taken_client]
save_file_name= f'../../Data/mnist/noisy/FedSTD_0.3_1_{client_percent}_{num_client}_{num_epochs}_{alpha}_{beta}_noise_mnist.pkl'
save_file(save_file_name,sample_list)