# implementing FL
import numpy as np

from tensorflow.keras import backend as K
from Code.utils.model import SimpleMLP2, SimpleMLP,test_model
from Code.utils.math_function import weight_scalling_factor,fed_avg_weight,scale_model_weights,sum_scaled_weights,weight_std_dev_median
import statistics

def FedPID_algo(comms_round, client_taken, clients_batched,x_test,y_test):
    limit_integral = 0
    limit_differential = 0
    # create optimizer
    lr = 0.01
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = 'adam'
    Global_accuracy_pid = list()
    client_accuracy_pid = list()
    Global_weight = list()
    Global_weight_pid = list()
    join_client_pid = list()
    local_weight_pid = list()
    join_client = list()
    # initialize global model
    smlp_global = SimpleMLP2()
    global_model = smlp_global.build(784)

    client_accuracy_STD = list()
    import time

    start_time = time.time()
    for comm_round in range(comms_round):

        # get the global model's weights - will serve as the initial weights for all local models
        global_weights = global_model.get_weights()

        # initial list to collect local model weights after scalling
        scaled_local_weight_list = list()
        scaled_global_weight_list = list()
        accuracy_list = list()
        # randomize client data - using keys
        client_names = list(clients_batched.keys())
        # random.shuffle(client_names)

        # loop through each client and create new local model
        for i in range(client_taken):
            smlp_local = SimpleMLP2()
            local_model = smlp_local.build(784)
            local_model.compile(loss=loss,
                                optimizer=optimizer,
                                metrics=metrics)

            # set local model weight to the weight of the global model
            local_model.set_weights(global_weights)

            # fit local model with client's data
            history = local_model.fit(clients_batched[client_names[i]], epochs=1, verbose=1)
            accuracy_list.append(history.history['accuracy'][-1])
            scaled_local_weight_list.append(local_model.get_weights())

            # clear session to free memory after each communication round
            K.clear_session()

        local_weight_pid.append(scaled_local_weight_list)
        client_accuracy_pid.append(accuracy_list)
        if comm_round > 11:
            Client_val = list()
            for b in range(client_taken):
                integral = 0
                for a in range(10):
                    integral += client_accuracy_pid[comm_round - a][b] - client_accuracy_pid[comm_round - a - 1][b]
                Client_val.append(integral)

            difference = np.subtract(client_accuracy_pid[comm_round], client_accuracy_pid[comm_round - 1])
            index_1 = np.where(difference > limit_integral)
            Client_val = np.array(Client_val)
            index_2 = np.where(Client_val > limit_differential)
            index = np.union1d(index_1, index_2)
            join_client.append(index)
            length = len(index)
            if length > 2:
                # proportion Section
                data = [accuracy_list[index[i]] for i in range(len(index))]
                val = np.zeros(len(data))
                for i in range(len(data)):
                    if abs(data[i] - statistics.median(data)) <= statistics.stdev(data) * 1:
                        val[i] = statistics.stdev(data) * .5 + .01
                    #         elif statistics.stdev(data)*1 <= abs(data[i] - statistics.median(data))<=statistics.stdev(data)*2:
                    #             val[i] = statistics.stdev(data)*1.5
                    else:
                        val[i] = abs(data[i] - statistics.median(data)) + .01
                weighted_value = [np.reciprocal(val[j]) / np.sum(np.reciprocal(val)) for j in range(len(val))]

                print(length)
                print(index)
                print(weighted_value)
                for x in range(length):
                    scaled_weights = scale_model_weights(scaled_local_weight_list[index[x]], weighted_value[x])
                    scaled_global_weight_list.append(scaled_weights)
                average_weights = sum_scaled_weights(scaled_global_weight_list)
                scaled_global = list()
                scale_factor = 1 / 5
                scaled_global.append(scale_model_weights(average_weights, scale_factor))
                for y in range(4):
                    scaled_global.append(scale_model_weights(Global_weight[comm_round - 1 - y], scale_factor))

                average_global_weights = sum_scaled_weights(scaled_global)

            else:
                print('none')
                average_weights = Global_weight[comm_round - 1]
                scaled_global = list()
                scale_factor = 1 / 5
                scaled_global.append(scale_model_weights(average_weights, scale_factor))
                for y in range(4):
                    scaled_global.append(scale_model_weights(Global_weight[comm_round - 1 - y], scale_factor))

                average_global_weights = sum_scaled_weights(scaled_global)
        else:
            for x in range(client_taken):
                scaling_factor = 1 / client_taken
                scaled_weights = scale_model_weights(scaled_local_weight_list[x], scaling_factor)
                scaled_global_weight_list.append(scaled_weights)
            average_global_weights = sum_scaled_weights(scaled_global_weight_list)

        Global_weight.append(average_global_weights)
        global_model.compile(loss=loss,
                             optimizer=optimizer,
                             metrics=metrics)

        # update global model
        global_model.set_weights(average_global_weights)

        score = global_model.evaluate(x_test, y_test, verbose=0)
        print('communication round:', comm_round)
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])
        Global_accuracy_pid.append(score[1])
        Global_weight_pid.append(average_global_weights)

    end_time = time.time()
    print('total time taken:', end_time - start_time)

    return Global_accuracy_pid,client_accuracy_pid, Global_weight_pid, local_weight_pid,join_client