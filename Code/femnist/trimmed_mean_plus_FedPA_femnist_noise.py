import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data_femnist, creating_shuffling_clients,create_client
from Code.algorithm.trimmed_mean_plus_FedPA_algo_femnist_sig import TM_plus_FedPA_algo_femnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 250
num_client= 3000
client_percent= .1
percent_trim = 0.1

file_name = "../../Data/Femnist/Dataset_noise_3000_0.3_1.pkl"
Dataset= open_file(file_name)

file_name = "../../Data/Femnist/Testset_3000.pkl"
Testset= open_file(file_name)


#process and batch the training data for each client
data = Dataset[0]


#process and batch the test set
bad_client= Dataset[1]
x_test= Testset[0]
y_test= Testset[1]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2  = TM_plus_FedPA_algo_femnist(num_epochs, num_client, data ,x_test,y_test, client_percent,percent_trim)
sample_list=[GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2 ]

save_file_name= f'../../Data/Femnist/sigmetrix/TM_plus_FedPA_batch_test.pkl'
save_file(save_file_name,sample_list)