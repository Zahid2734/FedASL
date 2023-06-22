# implementing FL
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from Code.utils.model import SimpleMLP4, SimpleMLP,test_model
from Code.utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median
import random
import math
import random
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(37)
random.seed(5)
tf.random.set_seed(8)

def fedzero_weight(client_taken,bad_client,client_percent):
    a= len(bad_client)
    bad_client= np.sort(bad_client)
    b= math.ceil(client_taken*client_percent)
    weight= 1/(b-a)
    weight_list=  np.full((b),weight)
    j=0
    if len(bad_client)>0:
        for i in range(b):
            if bad_client[j]==i:
                weight_list[i]=0
                if j<a-1:
                    j=j+1
    return weight_list,b



def FedZERO_algo_cifar10(comms_round, client_taken,bad_client, clients_batched,x_test,y_test, client_percent):
    lr = 0.01
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = 'adam'
    GLobal_accuracy = list()
    Client_accuracy = list()
    Global_weight = list()
    local_weight = list()
    taken_client= list()
    # initialize global model
    smlp_global = SimpleMLP4()
    global_model = smlp_global.build()
    # commence global training loop
    batch_size = 32
    import time


    for comm_round in range(comms_round):
        start_time = time.time()
        # randomlist = random.sample(range(0,client_taken), math.ceil(client_taken*client_percent))
        # get the global model's weights - will serve as the initial weights for all local models
        # taken_client.append(randomlist)
        global_weights = global_model.get_weights()
        weight_list, client_len=fedzero_weight(client_taken,bad_client,client_percent)
        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        accuracy_list = list()

        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        # random.shuffle(client_names)
        total_data = list()
        # loop through each client and create new local model
        for i in range(client_len):
            smlp_local = SimpleMLP4()
            local_model = smlp_local.build()
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # fit local model with client's data
            hist = local_model.fit(clients_batched[client_names[i]], epochs=1, verbose=1)
            #         test_loss, test_acc = local_model.evaluate(clients_batched[client_names[i]])
            #         print(test_acc)
            #         print(hist.history['accuracy'][-1])
            accuracy_list.append(hist.history['accuracy'][-1])
            scaled_local_weight_list.append(local_model.get_weights())

            # clear session to free memory after each communication round
            K.clear_session()

        # to get the average over all the local model, we simply take the sum of the scaled weights
        local_weight.append(scaled_local_weight_list)
        Client_accuracy.append(accuracy_list)
        weighted_value = weight_list
        print(weighted_value)
        scaled_weight = list()
        for k in range(client_len):
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

    return GLobal_accuracy,Client_accuracy, taken_client