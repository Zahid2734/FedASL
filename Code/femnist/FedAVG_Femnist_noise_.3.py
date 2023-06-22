import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data_femnist, creating_shuffling_clients,create_client
from Code.algorithm.FedAVG_algo_femnist_sig import FedAVG_algo_femnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 250
num_client= 3000
client_percent= .1

file_name = "../../Data/Femnist/Dataset_noise_3000_0.3_1.pkl"
Dataset= open_file(file_name)

file_name = "../../Data/Femnist/Testset_3000.pkl"
Testset= open_file(file_name)


data = Dataset[0]
#process and batch the test set
bad_client= Dataset[1]
x_test= Testset[0]
y_test= Testset[1]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

GLobal_accuracy,Client_accuracy, local_weight, taken_client  = FedAVG_algo_femnist(num_epochs, num_client, data,x_test,y_test, client_percent)
sample_list=[GLobal_accuracy,Client_accuracy,  taken_client]
save_file_name= f'../../Data/Femnist/sigmetrix/FedAVG_noise_femnist.pkl'
save_file(save_file_name,sample_list)