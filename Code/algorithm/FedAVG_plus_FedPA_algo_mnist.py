# implementing FL
import numpy as np
import keras
import random
import statistics
import pickle
import math

import tensorflow as tf

from tensorflow.keras import backend as K
from Code.utils.model import SimpleMLP2, SimpleMLP,test_model
from Code.utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median, weight_alpha_beta_median
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

def FedAVG_plus_FedPA_algo_mnist(comms_round, client_taken, clients_batched,x_test,y_test , client_percent):
    alpha= 1
    beta= .2
    lr = 0.01
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = 'adam'
    GLobal_accuracy_STD = list()
    Client_accuracy_STD = list()
    Global_weight_STD = list()
    local_weight_STD = list()
    scaled_weight_STD= list()
    taken_client= list()
    Filter1 = list()
    Filter2= list()
    passed_client_STD= list()
    previous_accuracy= list()
    batch_size = 32
    # initialize global model
    smlp_global = SimpleMLP2()
    global_model = smlp_global.build(784)
    # commence global training loop
    import time

    for comm_round in range(comms_round):
        start_time = time.time()
        randomlist = random.sample(range(0, client_taken), math.ceil(client_taken * client_percent))
        print(randomlist)
        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()
        taken_client.append(randomlist)
        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        scaled_global_weight_list = list()
        accuracy_list = list()
        prev_acc= list()
        passed_client= list()
        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        # random.shuffle(client_names)
        total_data = list()
        filter1_list=list()
        filter2_list=list()
        # loop through each client and create new local model
        for i in range(len(randomlist)):
            local_time = time.time()
            smlp_local = SimpleMLP2()
            local_model = smlp_local.build(784)
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)
            prefilter_time = time.time()
            local_score = local_model.evaluate(clients_batched[client_names[randomlist[i]]], verbose=0)
            pretrain_time = time.time() - prefilter_time
            print("time taken for pretrain", pretrain_time)
            prev_acc.append(local_score[1])
            # fit local model with client's data
            train_time = time.time()
            history = local_model.fit(clients_batched[client_names[randomlist[i]]], epochs=1, verbose=1)
            accuracy_list.append(history.history['accuracy'][-1])
            print("time_for local training ",  time.time() - train_time)

            if comm_round>9:

                if abs(history.history['accuracy'][-1] - local_score[1]) < 1.5 * standard_dev and (median - 1.5 * standard_dev) < local_score[1]:
                    print('passed')
                    data_points = len(clients_batched[client_names[randomlist[i]]]) * batch_size
                    total_data.append(data_points)
                    scaled_local_weight_list.append(local_model.get_weights())
                    passed_client.append(history.history['accuracy'][-1])
                else:
                    print('not allowed')

                if median - 1.5 * standard_dev >= local_score[1]:
                    filter1_list.append(randomlist[i])
                    print("filter1", randomlist[i])

                if abs(history.history['accuracy'][-1] - local_score[1]) >= 1.5 * standard_dev:
                    filter2_list.append(randomlist[i])
                    print("filter2", randomlist[i])

            else:
                data_points = len(clients_batched[client_names[randomlist[i]]]) * batch_size
                total_data.append(data_points)
                scaled_local_weight_list.append(local_model.get_weights())
                passed_client.append(history.history['accuracy'][-1])
            # clear session to free memory after each communication round
            local_complete = time.time() - local_time
            print('time taken in client_side and percetage of overhead added by pretrain', local_complete,
                  pretrain_time / local_complete)
            K.clear_session()
        Filter1.append(filter1_list)
        Filter2.append(filter2_list)
        previous_accuracy.append(prev_acc)
        local_weight_STD.append(scaled_local_weight_list)
        Client_accuracy_STD.append(accuracy_list)
        passed_client_STD.append(passed_client)
        acccuracy_list = np.array([passed_client])
        weighted_value_std, median, standard_dev= weight_alpha_beta_median(passed_client,alpha, beta)
        weighted_value = fed_avg_weight(total_data)
        print(weighted_value)
        print(f'median is {median} and Standard deviation {standard_dev}')
        scaled_weight_STD.append(weighted_value)
        scaled_weight = list()
        for k in range(len(passed_client)):
            scaled_weight.append(scale_model_weights(scaled_local_weight_list[k], weighted_value[k]))

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_weight)
        global_model.compile(loss=loss,
                             optimizer=optimizer,
                             metrics=metrics)

        # update global model
        global_model.set_weights(average_weights)
        score = global_model.evaluate(x_test, y_test, verbose=0)
        print('communication round:', comm_round)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        GLobal_accuracy_STD.append(score[1])
        Global_weight_STD.append(average_weights)
        print('total time taken:', time.time() - start_time)
    return GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, Filter1,Filter2