import pickle
import tensorflow as tf
from Code.utils.client_creation import batch_data
from Code.algorithm.FedPID_algo import FedPID_algo
from Code.utils.file_handler import save_file, open_file

# user will define
num_epochs= 100
num_client= 24

# file_name = "../Data/Dataset0.3_1_24_flip.pkl"
file_name = "../../../Data/mnist/Dataset0.3_1_24_shuffle.pkl"
Dataset= open_file(file_name)

#process and batch the training data for each client
bad_client= Dataset[1]
clients= Dataset[0]
clients_batched = dict()
for (client_name, data) in clients.items():
    clients_batched[client_name] = batch_data(data)

#process and batch the test set
x_test= Dataset[2]
y_test= Dataset[3]
test_batched = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(len(y_test))



Global_accuracy_pid,client_accuracy_pid, Global_weight_pid, local_weight_pid,join_client = FedPID_algo(num_epochs, num_client, clients_batched,x_test,y_test )
sample_list=[Global_accuracy_pid,client_accuracy_pid, Global_weight_pid, local_weight_pid,join_client, bad_client]
# save_file_name= f'../Data/mnist/flipping/FedPID_0.3_1_24_{num_epochs}_flip.pkl'
save_file_name= f'../../Data/mnist/shuffling/FedPID_0.3_1_24_{num_epochs}_shuffle.pkl'
save_file(save_file_name,sample_list)