import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
from Code.algorithm.FeAVG_algo_cifar10_sig import FedAVG_algo_cifar10
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 200
num_client= 100
client_percent= .3
file_name = "../../Data/cifar10/Dataset0.3_1_100_noise_cifar10.pkl"
Dataset= open_file(file_name)
data = Dataset[0]

#process and batch the test set
bad_client= Dataset[1]
x_test= Dataset[2]
y_test= Dataset[3]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

GLobal_accuracy,Client_accuracy, local_weight, taken_client= FedAVG_algo_cifar10(num_epochs, num_client, data,x_test,y_test, client_percent)

sample_list=[GLobal_accuracy,Client_accuracy,  taken_client]
save_file_name= f'../../Data/cifar10/sigmetrix/FedAVG_noise_cifar10.pkl'

save_file(save_file_name,sample_list)