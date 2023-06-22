
import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
from Code.algorithm.trimmed_mean_plus_FedPA_algo_mnist_sig import TM_plus_FedPA_algo_mnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 200
num_client= 100
client_percent= .3
percent_trim =.3

file_name = "../../Data/mnist/Dataset0.3_1_100_noise_mnist.pkl"
Dataset= open_file(file_name)

#process and batch the training data for each client
data= Dataset[0]



#process and batch the test set
bad_client= Dataset[1]
x_test= Dataset[2]
y_test= Dataset[3]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2 = TM_plus_FedPA_algo_mnist(num_epochs, num_client, data,x_test,y_test, client_percent, percent_trim)

sample_list=[GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2]
save_file_name= f'../../Data/mnist/sigmetrix/TM_plus_FedPA_batch_test.pkl'

save_file(save_file_name,sample_list)