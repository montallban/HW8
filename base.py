'''
Author: Andrew H. Fagg
Modified by: Michael Montalbano
'''


import tensorflow as tf
import pandas as pd
import numpy as np
import os
import fnmatch
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout
from tensorflow.keras import Sequential
import random
import re

import png
import sklearn.metrics
import os
import fnmatch
import re
import numpy as np
from create_unet import *
from chesapeake_data import *
from job_control import *
import sys

tf_tools = "../../../../../tf_tools/"
sys.path.append(tf_tools + "metrics")
sys.path.append(tf_tools + "networks")
sys.path.append(tf_tools + "experiment_control")

import argparse
import pickle

tf.config.threading.set_intra_op_parallelism_threads(8)

#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=18

plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE

plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE

#################################################################

file_base = '/scratch/fagg/chesapeake/'


#################################################################

def create_parser():
    # Parse the command-line arguments
    parser = argparse.ArgumentParser(description='BMI Learner', fromfile_prefix_chars='@')
    parser.add_argument('-network',type=str,default='unet',help="Choose shallow, deep, or inception")
    parser.add_argument('-rotation', type=int, default=0, help='Cross-validation rotation')
    parser.add_argument('-epochs', type=int, default=2000, help='Training epochs')
    parser.add_argument('-dataset', type=str, default=r"C:\Users\User\AML\HW7\data", help='Data set directory')
    parser.add_argument('-Ntraining', type=int, default=4, help='Number of training folds')
    parser.add_argument('-exp_index', type=int, default=0, help='Experiment index')
    parser.add_argument('-Nfolds', type=int, default=5, help='Maximum number of folds')
    parser.add_argument('-results_path', type=str, default='./results_hw7', help='Results directory')
    parser.add_argument('-hidden', nargs='+', type=int, default=[100, 5], help='Number of hidden units per layer (sequence of ints)')
    parser.add_argument('-conv_size', nargs='+', type=int, default=[3,5], help='Convolution filter size per layer (sequence of ints)')
    parser.add_argument('-conv_nfilters', nargs='+', type=int, default=[10,15], help='Convolution filters per layer (sequence of ints)')
    parser.add_argument('-pool', nargs='+', type=int, default=[2,2], help='Max pooling size (1=None)')
    parser.add_argument('-dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('-lrate', type=float, default=0.001, help="Learning rate")
    parser.add_argument('-L2_regularizer', '-l2', type=float, default=0.001, help="L2 regularization parameter")
    parser.add_argument('-min_delta', type=float, default=0.0, help="Minimum delta for early termination")
    parser.add_argument('-patience', type=int, default=100, help="Patience for early termination")
    parser.add_argument('-verbose', '-v', action='count', default=0, help="Verbosity level")
    parser.add_argument('-experiment_type', type=str, default="basic", help="Experiment type")
    parser.add_argument('-nogo', action='store_true', help='Do not perform the experiment')
    parser.add_argument('-batch_size', type=int, default=80, help="Patience for early termination")

    
    return parser

#################################################################
def check_args(args):
    assert (args.rotation >= 0 and args.rotation < args.Nfolds), "Rotation must be between 0 and Nfolds"
    assert (args.Ntraining >= 1 and args.Ntraining <= (args.Nfolds-1)), "Ntraining must be between 1 and Nfolds-2"
    assert (args.dropout is None or (args.dropout > 0.0 and args.dropout < 1)), "Dropout must be between 0 and 1"
    assert (args.lrate > 0.0 and args.lrate < 1), "Lrate must be between 0 and 1"
    assert (args.L2_regularizer is None or (args.L2_regularizer > 0.0 and args.L2_regularizer < 1)), "L2_regularizer must be between 0 and 1"
    
def augment_args(args):
    '''
    Use the jobiterator to override the specified arguments based on the experiment index. 
    @return A string representing the selection of parameters to be used in the file name
    '''
    print("augmenting args")
    index = args.exp_index
    if(index == -1):
        return ""
    
    # Create parameter sets to execute the experiment on.  This defines the Cartesian product
    #  of experiments that we will be 
    if args.experiment_type is None:
        return ""
    elif args.experiment_type == "basic":
        print("basic")
        p = {'rotation': range(5)}
    elif args.experiment_type == "test":
        print("test")
        p = {'L2_regularizer': [None, 0.0001, 0.001, 0.005, 0.01],
             'dropout': [None, 0.1, 0.2, 0.3, 0.4], 
             'rotation': range(5)}
    else:
        assert False, "Bad experiment type"
        
    
    # Create the iterator
    ji = JobIterator(p)
    print("Total jobs:", ji.get_njobs())
    
    # Check bounds
    assert (args.exp_index >= 0 and args.exp_index < ji.get_njobs()), "exp_index out of range"

    # Print the parameters specific to this exp_index
    print(ji.get_index(args.exp_index))
    
    # Push the attributes to the args object and return a string that describes these structures
    return ji.set_attributes_by_index(args.exp_index, args)
 
    
#################################################################

def generate_fname(args, params_str):
    '''
    Generate the base file name for output files/directories.
    
    The approach is to encode the key experimental parameters in the file name.  This
    way, they are unique and easy to identify after the fact.
    :param args: from argParse
    :params_str: String generated by the JobIterator
    '''    
    # Dropout
    if args.dropout is None:
        dropout_str = 'None'
    else:
        dropout_str = 'drop_%0.2f_'%(args.dropout)
        
    # L2 regularization
    if args.L2_regularizer is None:
        regularizer_str = 'None'
    else:
        regularizer_str = 'L2_%0.6f_'%(args.L2_regularizer)


        
    # Put it all together, including #of training folds and the experiment rotation
    if(args.network=="recurrent"):
        return "%s/%s_recurrent_dropout_%s_l2_%s_ntrain_%02d_rot_%02d"%(args.results_path, args.experiment_type,
                                                                                          regularizer_str,
                                                                                          dropout_str,
                                                                                          args.Ntraining,
                                                                                          args.rotation)       


def training_set_generator_images(ins, outs, batch_size=10,
                          input_name='input', 
                        output_name='output'):
    '''
    Generator for producing random minibatches of image training samples.
    
    @param ins Full set of training set inputs (examples x row x col x chan)
    @param outs Corresponding set of sample (examples x nclasses)
    @param batch_size Number of samples for each minibatch
    @param input_name Name of the model layer that is used for the input of the model
    @param output_name Name of the model layer that is used for the output of the model
    '''
    
    while True:
        # Randomly select a set of example indices
        example_indices = random.choices(range(ins.shape[0]), k=batch_size)
        
        # The generator will produce a pair of return values: one for inputs and one for outputs
        yield({input_name: ins[example_indices,:,:,:]},
             {output_name: outs[example_indices,:]})

#################################################################
def execute_exp(args=None):
    '''
    Perform the training and evaluation for a single model
    
    @args Argparse arguments
    '''
    # Check the arguments

    if args is None:
        # Case where no args are given (usually, because we are calling from within Jupyter)
        #  In this situation, we just use the default arguments
        parser = create_parser()
        args = parser.parse_args([])
        
    # Modify the args in specific situations
    #augment_args(args)

    args_str = augment_args(args)
    
    # Perform the experiment
    if(args.nogo):
        print("nogo")
        # No!
        return
    
    # Load data
    if args.rotation != 0:
        sys.exit()
    fold = "F" + str(args.rotation)
    # specify half 
    print('loading')
    ins_train, mask, outs_train, weights = load_files_from_dir(args.dataset + '\\train\\' + fold, filt='-*[02]?')
    print(outs_train.shape)

    [new_outs, labels] = pixel_class_to_image_class(outs_train)
    new_outs = new_outs[:,4]
    print(new_outs.shape)
    outs_train = np.reshape(new_outs, (len(new_outs), -1))
    print(outs_train.shape)

    ins_val, mask, outs_val, weights = load_files_from_dir(args.dataset + '\\train\\' + fold, filt='-[12345]9?')
    [new_outs, labels] = pixel_class_to_image_class(outs_val)
    outs_val = np.reshape(new_outs, (len(new_outs), -1))

    print("finished loading")
    if args.network == 'unet':
        model = create_uNet(ins_train.shape[1:4], nclasses=1)

    print(model.summary())

    ins_test, mask, outs_test, weights = load_files_from_dir(args.dataset+ '\\valid\\' + fold, filt='-*?')
    [new_outs, labels] = pixel_class_to_image_class(outs_test)
    outs_test = np.reshape(new_outs, (len(new_outs), -1))

    # fit model
    # model.fit(ins,outs)
    generator = training_set_generator_images(ins_train, outs_train, batch_size=args.batch_size,
                            input_name='input',
                            output_name='output')

    early_stopping_cb = keras.callbacks.EarlyStopping(patience=args.patience,
                                                      restore_best_weights=True,
                                                      min_delta=args.min_delta)

    # Learn
    history = model.fit(x=generator, 
                        epochs=args.epochs, 
                        steps_per_epoch=10,
                        use_multiprocessing=False, 
                        verbose=args.verbose>=2,
                        validation_data=(ins_val, outs_val), 
                        callbacks=[early_stopping_cb])

    #print("val eval",model.evaluate(ins_val, outs_val))
    #print("test eval", model.evaluate(ins_val, outs_val))
    # Generate log data
    results = {}
    results['args'] = args
    results['predict_training'] = model.predict(ins_train)
    results['predict_training_eval'] = model.evaluate(ins_train, outs_train)
    results['true_training'] = outs_train
    results['predict_validation'] = model.predict(ins_val)
    results['predict_validation_eval'] = model.evaluate(ins_val, outs_val)
    results['true_validation'] = outs_val
    results['predict_testing'] = model.predict(ins_test)
    results['predict_testing_eval'] = model.evaluate(ins_test, outs_test)
    #results['folds'] = folds
    results['history'] = history.history
    
    # Save results
    fbase = generate_fname(args, args_str)
    results['fname_base'] = fbase
    fp = open("%s_results.pkl"%(fbase), "wb")
    pickle.dump(results, fp)
    fp.close()
    
    # Model
    model.save("%s_model"%(fbase))
    
    return model
#################################################################
if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)
    execute_exp(args)
