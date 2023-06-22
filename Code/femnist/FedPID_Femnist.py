import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data_femnist, creating_shuffling_clients,create_client
from Code.algorithm.FedPID_algo_Femnist import FedPID_algo_Femnist
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 2
num_client= 50

file_name = "../../../Data/Femnist/Dataset_clean_50_100.pkl"
Dataset= open_file(file_name)

#process and batch the training data for each client
clients= Dataset[0]
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data_femnist(data)

#process and batch the test set
# bad_client= Dataset[1]
x_test= Dataset[1]
y_test= Dataset[2]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))

Global_accuracy_pid,client_accuracy_pid, Global_weight_pid, local_weight_pid,join_client = FedPID_algo_Femnist(num_epochs, num_client, clients_batched,x_test,y_test)