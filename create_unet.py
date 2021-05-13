# Author: Michael Montalbano
# Create U-Network or Autoencoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D
from tensorflow.keras.models import Model


# channels should increase as you step down the U-Net
# ratio of rows*columns/channels should go down
def create_uNet(input_shape, nclasses, filters=[30,45,60], 
                   lambda_regularization=None, activation='elu'):
    if lambda_regularization is not None:
        lambda_regularization = keras.regularizers.l2(lambda_regularization)
    
    tensor_list = []
    input_tensor = Input(shape=input_shape, name="input")

    # you could make a loop
    # to append and pop



    # 256x256
        
    tensor = Convolution2D(filters[0],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(input_tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    #############################
    tensor_list.append(tensor)
   
    tensor = MaxPooling2D(pool_size=(2,2),
                          strides=(2,2),
                          padding='same')(tensor)
    
    # 128x128

    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor_list.append(tensor)

    tensor = MaxPooling2D(pool_size=(2,2),
                          strides=(2,2),
                          padding='same')(tensor)

    # 64x64

    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    # upsample

    # 128x128
    tensor = UpSampling2D(size=2) (tensor) # take 1 pixel and expand it out to 2 x 2

    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Concatenate()([tensor, tensor_list.pop()])

    # upsample
    
    tensor = UpSampling2D(size=2) (tensor) # take 1 pixel and expand it out to 2 x 2
    
    # 256 x 256
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Concatenate()([tensor, tensor_list.pop()])

    #############################        
    output_tensor = Convolution2D(nclasses,
                          kernel_size=(1,1),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation='sigmoid',name='output')(tensor)
    

    model = Model(inputs=input_tensor, outputs=output_tensor)
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, 
                                    epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt,
                 metrics=["binary_accuracy"])
    
    return model

def create_seq(input_shape, nclasses, filters=[30,45,60], 
                   lambda_regularization=None, activation='elu'):
    # A sequential model for semantic reasoning

    if lambda_regularization is not None:
        lambda_regularization = keras.regularizers.l2(lambda_regularization)
    
    tensor_list = []
    input_tensor = Input(shape=input_shape, name="input")

    # 256x256
        
    tensor = Convolution2D(filters[0],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(input_tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    


    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    #############################        
    output_tensor = Convolution2D(nclasses,
                          kernel_size=(1,1),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation='sigmoid',name='output')(tensor)
    

    model = Model(inputs=input_tensor, outputs=output_tensor)
    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, 
                                    epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt,
                 metrics=["binary_accuracy"])
    
    return model
