# implementing FL
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from Code.utils.model import SimpleMLP3, SimpleMLP,test_model
from Code.utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median
from Code.utils.client_creation import batch_data_femnist, creating_shuffling_clients,create_client
import random
import math
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

def FedAVG_algo_femnist(comms_round, client_taken, data,x_test,y_test, client_percent):
    lr = 0.01
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metrics = ['accuracy']
    optimizer = 'adam'
    GLobal_accuracy = list()
    Client_accuracy = list()
    Global_weight = list()
    local_weight = list()
    taken_client= list()
    # initialize global model
    smlp_global = SimpleMLP3()
    global_model = smlp_global.build(784)
    # commence global training loop
    batch_size = 32
    client_names = list(data.keys())
    import time


    for comm_round in range(comms_round):
        start_time = time.time()
        randomlist = random.sample(range(0,client_taken), math.ceil(client_taken*client_percent))
        # get the global model's weights - will serve as the initial weights for all local models
        taken_client.append(randomlist)
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        accuracy_list = list()

        # randomize client data - using keys
        # random.shuffle(client_names)
        total_data = list()
        # loop through each client and create new local model
        for i in range(len(randomlist)):
            client_time= time.time()
            # data_points = 500

            smlp_local = SimpleMLP3()
            local_model = smlp_local.build(784)
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)
            data_points = len(data[client_names[randomlist[i]]])
            total_data.append(data_points)
            # fit local model with client's data
            train_data = {}
            print(client_names[randomlist[i]])
            train_data[client_names[randomlist[i]]] = batch_data_femnist(data[client_names[randomlist[i]]])
            history = local_model.fit(train_data[client_names[randomlist[i]]], epochs=1, verbose=1)
            scaled_local_weight_list.append(local_model.get_weights())

            # clear session to free memory after each communication round
            print('time taken by client' , time.time()-client_time)
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        local_weight.append(scaled_local_weight_list)
        Client_accuracy.append(accuracy_list)
        weighted_value = fed_avg_weight(total_data)
        scaled_weight = list()
        for k in range(len(randomlist)):
            scaled_weight.append(scale_model_weights(scaled_local_weight_list[k], weighted_value[k]))

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_weight)

        global_model.compile(loss=loss,
                             optimizer=optimizer,
                             metrics=metrics)

        # update global model
        global_model.set_weights(average_weights)
        Global_weight.append(average_weights)
        score = global_model.evaluate(x_test, y_test, verbose=0)
        print('communication round:', comm_round)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        GLobal_accuracy.append(score[1])

        print('total time taken:', time.time() - start_time)

    return GLobal_accuracy,Client_accuracy, local_weight, taken_client