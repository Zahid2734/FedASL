# implementing FL
import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data
from Code.algorithm.median_algo_cifar10 import median_algo
from Code.utils.file_handler import save_file, open_file



# user will define
num_epochs= 200
num_client= 100
alpha= 1
beta= .2
client_percent= .3
file_name = "../../Data/cifar10/Dataset0.3_0.75_100_shuffle_cifar10.pkl"
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
GLobal_accuracy,Client_accuracy, taken_client= median_algo(num_epochs, num_client, clients_batched,x_test,y_test, client_percent)
sample_list=[GLobal_accuracy,Client_accuracy, taken_client, bad_client, taken_client]
save_file_name= f'../../Data/cifar10/data percent/new/median_.75.pkl'
save_file(save_file_name,sample_list)