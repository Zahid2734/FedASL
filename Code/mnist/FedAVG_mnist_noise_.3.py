import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
from Code.algorithm.FedAVG_algo_sig import FedAVG_algo_mnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 200
num_client= 100
client_percent= .3

file_name = "../../Data/mnist/Dataset0.3_1_100_noise_mnist.pkl"
Dataset= open_file(file_name)

data = Dataset[0]

#process and batch the test set
bad_client= Dataset[1]
x_test= Dataset[2]
y_test= Dataset[3]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

GLobal_accuracy,Client_accuracy, local_weight, taken_client= FedAVG_algo_mnist(num_epochs, num_client, data,x_test,y_test, client_percent)

sample_list=[GLobal_accuracy,Client_accuracy,  taken_client]
save_file_name= f'../../Data/mnist/sigmetrix/FedAVG_noise_cifar10.pkl'

save_file(save_file_name,sample_list)