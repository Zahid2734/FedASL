# implementing FL
import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data
from Code.algorithm.FedZERO_algo import FedZERO_algo_cifar10
from Code.utils.file_handler import save_file, open_file



# user will define
num_epochs= 200
num_client= 10
client_percent= 1
file_name = "../../Data/cifar10/Dataset0.4_1_10_shuffle_cifar10.pkl"
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
null_client=[]


GLobal_accuracy_STD,Client_accuracy_STD,scaled_weight_STD= FedZERO_algo_cifar10(num_epochs, num_client,null_client, clients_batched,x_test,y_test,client_percent)

sample_list=[GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, bad_client]
save_file_name= f'../../Data/cifar10/shuffling/FedAVG 0.4 for barplot.pkl'
save_file(save_file_name,sample_list)