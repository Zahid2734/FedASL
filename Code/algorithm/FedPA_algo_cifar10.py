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
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

def FedPA_algo_cifar10(comms_round, client_taken, clients_batched,x_test,y_test , client_percent, alpha, beta, previous_average):
    alpha= alpha
    beta= beta
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
    passed_client_STD= list()
    previous_accuracy= list()
    # initialize global model
    smlp_global = SimpleMLP4()
    global_model = smlp_global.build()
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

        # loop through each client and create new local model
        for i in range(len(randomlist)):
            smlp_local = SimpleMLP4()
            local_model = smlp_local.build()
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)
            local_score= local_model.evaluate(clients_batched[client_names[randomlist[i]]], verbose=0)
            print(local_score[1])
            prev_acc.append(local_score[1])
            # fit local model with client's data
            history = local_model.fit(clients_batched[client_names[randomlist[i]]], epochs=1, verbose=1)
            accuracy_list.append(history.history['accuracy'][-1])
            print('difference of value for: ',{history.history['accuracy'][-1]-local_score[1]})

            if comm_round>9:
                if abs(history.history['accuracy'][-1]-local_score[1])< .15 and median -.05<local_score[1]:
                    print('passed')
                    scaled_local_weight_list.append(local_model.get_weights())
                    passed_client.append(history.history['accuracy'][-1])
                else:
                    print('not allowed')

            else:
                scaled_local_weight_list.append(local_model.get_weights())
                passed_client.append(history.history['accuracy'][-1])
            # clear session to free memory after each communication round
            K.clear_session()
        previous_accuracy.append(prev_acc)
        local_weight_STD.append(scaled_local_weight_list)
        Client_accuracy_STD.append(accuracy_list)
        passed_client_STD.append(passed_client)
        acccuracy_list = np.array([passed_client])
        weighted_value, median, standard_dev= weight_alpha_beta_median(passed_client,alpha, beta)
        print(weighted_value)
        print(f'median is {median} and Standard deviation {standard_dev}')
        scaled_weight_STD.append(weighted_value)
        scaled_weight = list()
        for k in range(len(passed_client)):
            scaled_weight.append(scale_model_weights(scaled_local_weight_list[k], weighted_value[k]))

        # to get the average over all the local model, we simply take the sum of the scaled weights
        average_weights = sum_scaled_weights(scaled_weight)
        scaled_global = list()
        if (comm_round> previous_average+1) and (previous_average>1):
            scale_factor = 1 / previous_average
            scaled_global.append(scale_model_weights(average_weights, scale_factor))
            for y in range(previous_average-1):
                scaled_global.append(scale_model_weights(Global_weight_STD[comm_round - 1 - y], scale_factor))

            average_global_weights = sum_scaled_weights(scaled_global)
        else:
            average_global_weights= average_weights


        global_model.compile(loss=loss,
                             optimizer=optimizer,
                             metrics=metrics)

        # update global model
        # update global model
        global_model.set_weights(average_global_weights)
        score = global_model.evaluate(x_test, y_test, verbose=0)
        print('communication round:', comm_round)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        GLobal_accuracy_STD.append(score[1])
        Global_weight_STD.append(average_global_weights)
        print('total time taken:', time.time() - start_time)
    return GLobal_accuracy_STD,Client_accuracy_STD, scaled_weight_STD, taken_client, passed_client_STD, previous_accuracy