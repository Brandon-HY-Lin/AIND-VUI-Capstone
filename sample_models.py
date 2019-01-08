from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, Conv2D, Dense, Input, 
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, Dropout)
from keras.layers.core import Reshape
import tensorflow as tf
from keras.layers import Lambda

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(units, activation=activation,
        return_sequences=True, implementation=2, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        output_length = input_length - dilated_filter_size + 1
    return (output_length + stride - 1) // stride


def deep_cnn_output_length(input_length, filter_size, border_mode, stride,
                           layer_cnn, dilation=1):
    
    prev_length = input_length
    
    for _ in range(layer_cnn):
        output_length = cnn_output_length(prev_length, filter_size, border_mode, stride, dilation)
        prev_length = output_length
        
    return output_length


def deep_rnn_model(input_dim, units, recur_layers, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    prev_rnn_out = input_data
    
    for index in range(recur_layers):
        rnn = GRU(units=units, implementation=2, return_sequences=True, name='GRU_{}'.format(index))(prev_rnn_out)
        bn_rnn = BatchNormalization(name='BatchNorm_{}'.format(index))(rnn)
        
        prev_rnn_out = bn_rnn
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn = Bidirectional(GRU(units=units, 
                                  implementation=2, 
                                  return_sequences=True, 
                                  name='GRU')
                             )(input_data)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def final_model():
    """ Build a deep network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Specify the layers in your network
    ...
    # TODO: Add softmax activation layer
    y_pred = ...
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    model.output_length = ...
    print(model.summary())
    return model


def cnn2d_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    shape_input = tf.shape(input_data)
    
#     print(shape_input)
    
    input_reshaped = tf.reshape(input_data, [shape_input[0], shape_input[1], shape_input[2], 1])
    
#     print(tf.shape(input_reshaped))
    
    # Add convolutional layer
    conv_2d = Conv2D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv2d')(input_reshaped)
    
#     print(tf.shape(conv_2d))
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_2d')(conv_2d)
    
    shape_bn_cnn = tf.shape(bn_cnn)
    
    bn_cnn_reshaped = tf.reshape(bn_cnn, [shape_bn_cnn[0], shape_bn_cnn[1], -1])
    
#     print(tf.shape(bn_cnn_reshaped))
    
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn_reshaped)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def squeeze_middle2axes_operator( x4d ) :
    shape = tf.shape( x4d ) # get dynamic tensor shape
#     x3d = tf.reshape( x4d, [shape[0], shape[1] * shape[2], shape[3] ] )
    x3d = tf.reshape( x4d, [shape[0], shape[1], shape[2] * shape[3] ] )

    return x3d


def squeeze_middle2axes_shape( x4d_shape ) :
    in_batch, in_rows, in_cols, in_filters = x4d_shape
    if ( None in [ in_rows, in_cols] ) :
#         output_shape = ( in_batch, None, in_filters )
        output_shape = ( in_batch, in_rows, in_cols * in_filters )
    else :
#         output_shape = ( in_batch, in_rows * in_cols, in_filters )
        output_shape = ( in_batch, in_rows, in_cols * in_filters )

    return output_shape


def cnn2d_rnn_model_2(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim, 1))
    
#     shape_input = tf.shape(input_data)
    
#     print(shape_input)
    
#     input_reshaped = tf.reshape(input_data, [shape_input[0], shape_input[1], shape_input[2], 1])
    
#     print(tf.shape(input_reshaped))
    
    # Add convolutional layer
    conv_2d = Conv2D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv2d')(input_data)
    
    print(tf.shape(conv_2d))
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_2d')(conv_2d)
    
    shape_bn_cnn = tf.shape(bn_cnn)
    
    bn_cnn_reshaped = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( bn_cnn )
    
#     bn_cnn_reshaped = tf.reshape(bn_cnn, [shape_bn_cnn[1], shape_bn_cnn[2], -1])
    
#     print(tf.shape(bn_cnn_reshaped))
    
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn_reshaped)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn2d_rnn_dropout_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim, 1))
    
    # Add convolutional layer
    conv_2d = Conv2D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv2d')(input_data)
    
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_2d')(conv_2d)
    
    shape_bn_cnn = tf.shape(bn_cnn)
    
    bn_cnn_reshaped = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( bn_cnn )
    
    drop_cnn = Dropout(dropout)(bn_cnn_reshaped)
    
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, dropout=dropout, name='rnn')(drop_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def deep_cnn2d_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, layer_cnn, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim, 1))
    
#     shape_input = tf.shape(input_data)

#     print(shape_input)
    
#     input_reshaped = tf.reshape(input_data, [shape_input[0], shape_input[1], shape_input[2], 1])
    
#     print(tf.shape(input_reshaped))
    
    cnn_in = input_data
    for index in range(layer_cnn):
        # Add convolutional layer
        conv_2d = Conv2D(filters * (index+1), kernel_size, 
                         strides=conv_stride, 
                         padding=conv_border_mode,
                         activation='relu',
                         name='conv2d_{}'.format(index))(cnn_in)

    #     print(tf.shape(conv_2d))

        # Add batch normalization
        bn_cnn = BatchNormalization(name='bn_conv_{}'.format(index))(conv_2d)
        
        cnn_in = bn_cnn
    
    shape_bn_cnn = tf.shape(bn_cnn)
    
    bn_cnn_reshaped = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( bn_cnn )
    
#     bn_cnn_reshaped = tf.reshape(bn_cnn, [shape_bn_cnn[1], shape_bn_cnn[2], -1])
    
#     print(tf.shape(bn_cnn_reshaped))
    
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn_reshaped)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: deep_cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, layer_cnn)
    print(model.summary())
    return model

def cnn_rnn_dropout_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, dropout, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    
    drop_cnn = Dropout(dropout)(bn_cnn)
    
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(drop_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    
    drop_rnn = Dropout(dropout)(bn_rnn)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(drop_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model


def deep_cnn2d_rnn_dropout_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, layer_cnn, dropout, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim, 1))
    
    cnn_in = input_data
    for index in range(layer_cnn):
        # Add convolutional layer
        conv_2d = Conv2D(filters * (index+1), kernel_size, 
                         strides=conv_stride, 
                         padding=conv_border_mode,
                         activation='relu',
                         name='conv2d_{}'.format(index))(cnn_in)

    #     print(tf.shape(conv_2d))

        # Add batch normalization
        bn_cnn = BatchNormalization(name='bn_conv_{}'.format(index))(conv_2d)
        
        drop_cnn = Dropout(dropout)(bn_cnn)
        
        cnn_in = drop_cnn
    
    
    shape_bn_cnn = tf.shape(bn_cnn)
    bn_cnn_reshaped = Lambda( squeeze_middle2axes_operator, output_shape = squeeze_middle2axes_shape )( bn_cnn )
    
    # Add a recurrent layer
    simp_rnn = GRU(units, activation='relu',
        return_sequences=True, implementation=2, dropout=dropout, name='rnn')(bn_cnn_reshaped)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization(name='BatchNorm')(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim, name='Dense'))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: deep_cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride, layer_cnn)
    print(model.summary())
    return model
