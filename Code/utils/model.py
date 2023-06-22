import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from keras.regularizers import L1L2

class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

class SimpleMLP2:
    @staticmethod
    def build(shape):
        model = Sequential()
        model.add(Dense(10,  activation='softmax',kernel_regularizer=L1L2(l1=0.01, l2=0.01),input_dim=shape))
        return model

class SimpleMLP3:
    @staticmethod
    def build(shape):
        model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_dim=shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10)
        ])
        return model


class SimpleMLP4:
    @staticmethod
    def build():
        model = Sequential()
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                         input_shape=(32, 32, 3)))
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Dropout(0.2))

        model.add(tf.keras.layers.Flatten())
        model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(Dense(10, activation='softmax'))
        return model


def test_model(X_test, Y_test,  model, comm_round):
    cce = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
    #logits = model.predict(X_test, batch_size=100)
    logits = model.predict(X_test)
    loss = cce(Y_test, logits)
    acc = accuracy_score(tf.argmax(logits, axis=1), tf.argmax(Y_test, axis=1))
    print('comm_round: {} | global_acc: {:.3%} | global_loss: {}'.format(comm_round, acc, loss))
    return acc, loss
