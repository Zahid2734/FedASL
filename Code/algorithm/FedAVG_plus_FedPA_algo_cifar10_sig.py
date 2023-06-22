# implementing FL
import numpy as np
import keras
import random
import statistics
import pickle
import math

import tensorflow as tf

from tensorflow.keras import backend as K
from Code.utils.model import SimpleMLP4, SimpleMLP,test_model
from Code.utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median, weight_alpha_beta_median
from Code.utils.client_creation import batch_data, creating_shuffling_clients,create_client
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

class AccuracyHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.Accuracy = []

    def on_batch_end(self, batch, logs={}):
        self.Accuracy.append(logs.get('accuracy'))

def FedAVG_plus_FedPA_algo_cifar10(comms_round, client_taken, data ,x_test,y_test , client_percent):
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
    smlp_global = SimpleMLP4()
    global_model = smlp_global.build()
    # commence global training loop
    import time
    client_names = list(data.keys())

    for comm_round in range(comms_round):
        start_time = time.time()
        randomlist = random.sample(range(0, client_taken-1), math.ceil(client_taken * client_percent))
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
        # random.shuffle(client_names)
        total_data = list()
        filter1_list=list()
        filter2_list=list()
        # loop through each client and create new local model
        for i in range(len(randomlist)):
            local_time = time.time()
            smlp_local = SimpleMLP4()
            local_model = smlp_local.build()
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            train_data = {}
            print(client_names[randomlist[i]])
            train_data[client_names[randomlist[i]]] = batch_data(data[client_names[randomlist[i]]])
            history = AccuracyHistory()
            local_model.fit(train_data[client_names[randomlist[i]]], epochs=1, verbose=1, callbacks=[history])
            print(history.Accuracy)
            post_accurcy = history.Accuracy[-1]
            pre_accuracy = history.Accuracy[0]
            accuracy_list.append(post_accurcy)
            if comm_round > 9:

                if abs(post_accurcy - pre_accuracy) < .15 and (median - .05) < pre_accuracy:
                    print('passed')
                    data_points = 500
                    total_data.append(data_points)
                    scaled_local_weight_list.append(local_model.get_weights())
                    passed_client.append(post_accurcy)
                else:
                    print('not allowed')

                if median -.05 >= pre_accuracy:
                    filter1_list.append(randomlist[i])
                    print("filter1", randomlist[i])

                if (post_accurcy - pre_accuracy) >=.15:
                    filter2_list.append(randomlist[i])
                    print("filter2", randomlist[i])


            else:
                data_points = 500
                total_data.append(data_points)
                scaled_local_weight_list.append(local_model.get_weights())
                passed_client.append(post_accurcy)
            # clear session to free memory after each communication round
            local_complete = time.time() - local_time
            K.clear_session()
        print('filter1 list',filter1_list)
        print('filter2 list', filter2_list)
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