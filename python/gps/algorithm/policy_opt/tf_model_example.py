""" This file provides an example tensorflow network used to define a policy. """

import tensorflow as tf


def init_weights(shape, name=None):
    return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)


def init_bias(shape, name=None):
    return tf.Variable(tf.zeros(shape, dtype='float'), name=name)


def batched_matrix_vector_multiply(vector, matrix):
    vector_batch_as_matricies = tf.expand_dims(vector, [1])
    mult_result = tf.batch_matmul(vector_batch_as_matricies, matrix)
    squeezed_result = tf.squeeze(mult_result, [1])
    return squeezed_result


def euclidean_loss_layer(a, b, precision):
    uP = batched_matrix_vector_multiply(a-b, precision)
    uPu = tf.reduce_sum(uP*(a-b), [1])
    loss = tf.reduce_mean(uPu)
    return loss


def get_input_layer(dim_input, dim_output):
    net_input = tf.placeholder("float", [None, dim_input], name='nn_input')
    action = tf.placeholder('float', [None, dim_output], name='action')
    precision = tf.placeholder('float', [None, dim_output, dim_output], name='precision')
    return net_input, action, precision


def get_mlp_layers(mlp_input, number_layers, dimension_hidden):
    cur_top = mlp_input
    for layer_step in range(0, number_layers):
        in_shape = cur_top.get_shape().dims[1].value
        cur_weight = init_weights([in_shape, dimension_hidden[layer_step]], name='w_' + str(layer_step))
        cur_bias = init_bias([dimension_hidden[layer_step]], name='b_' + str(layer_step))
        if layer_step != number_layers:  # final layer has no RELU
            cur_top = tf.nn.relu(tf.matmul(cur_top, cur_weight) + cur_bias)
        else:
            cur_top = tf.matmul(cur_top, cur_weight) + cur_bias

    return cur_top


def get_loss_layer(mlp_out, action, precision):
    return euclidean_loss_layer(a=action, b=mlp_out, precision=precision)
    #  out = (action - mlp_out)'*precision*(action-mlp_out)
    #  (u-uhat)'*A*(u-uhat)


def example_tf_network(dim_input=27, dim_output=7, batch_size=25):
    """
    An example of how one might want to specify a network in tensorflow.

    Args:
        dim_input: Dimensionality of input.
        dim_output: Dimensionality of the output.
        batch_size: Batch size.
    Returns:
        a dictionary containing inputs, outputs, and the loss function representing scalar loss.
    """
    n_layers = 3
    dim_hidden = (n_layers - 1) * [42]
    dim_hidden.append(dim_output)

    nn_input, action, precision = get_input_layer(dim_input, dim_output)
    mlp_applied = get_mlp_layers(nn_input, n_layers, dim_hidden)
    loss_out = get_loss_layer(mlp_out=mlp_applied, action=action, precision=precision)
    return_dict = {'inputs': [nn_input, action, precision], 'outputs': [mlp_applied], 'loss': [loss_out]}
    return return_dict
