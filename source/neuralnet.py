import utilities
import tensorflow as tf
from tensorflow import keras

# class neural_net():
#     def __int__(self, ):


def model_inputs():
    inputs = tf.compat.v1.placeholder(tf.int32, [None, None], name="input")
    targets = tf.compat.v1.placeholder(tf.int32, [None, None], name="target")
    lr = tf.compat.v1.placeholder(tf.float32, name="learning_rate")
    keep_prob = tf.compat.v1.placeholder(tf.float32, name="keep_prob")
    return inputs, targets, lr, keep_prob

def preprocess_targets(targets, word2int, batch_size):
    left_side = tf.fill([batch_size, 1], word2int["<SOS>"])
    right_side = tf.strided_slice(targets, [0,0], [batch_size, -1], [1,1])
    preprocessed_targets = tf.concat([left_side, right_side], axis=1)
    return preprocessed_targets

def encoder_rnn_layer(rnn_inputs, rnn_size, num_layers, keep_prob, sequence_length):
    lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm = tf.keras.L
    lstm_dropout = tf.contrib.rnn.DropoutWrapper(lstm, input_keep_prob = keep_prob)
