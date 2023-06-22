import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
from Code.algorithm.FedAVG_algo import FedAVG_algo
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 200
num_client= 100
client_percent= .3

file_name = "../../Data/mnist/Dataset0.4_1_100_flip_mnist.pkl"
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

GLobal_accuracy,Client_accuracy, taken_client = FedAVG_algo(num_epochs, num_client, clients_batched,x_test,y_test, client_percent)

sample_list=[GLobal_accuracy,Client_accuracy, taken_client, bad_client]
save_file_name= f'../../Data/mnist/flipping/FedAVG_0.4_1_{client_percent}_{num_client}_{num_epochs}_flip_mnist.pkl'

save_file(save_file_name,sample_list)