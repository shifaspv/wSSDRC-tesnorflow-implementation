from __future__ import print_function

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.

import os
import logging
import numpy as np
import tensorflow as tf
from model import Wavenet
from lib.optimizers import get_learning_rate, get_optimizer
from lib.audio_conditions_io import AudioConditionsReader
from lib.model_io import get_configuration, setup_logger, get_model_id
from lib.model_io import restore_variables
from lib.util import compute_receptive_field_length


cfg, learning_rate_params, optim_params, gc = get_configuration('train')

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["CUDA_VISIBLE_DEVICES"]

if cfg['model_id'] is not None:
    model_id = cfg['model_id']
else:
    model_id = get_model_id(cfg)

msg_logging_dir = os.path.join(cfg['base_dir'], cfg['logging_dir'], 'log_'+str(model_id)+'.txt') 
setup_logger('msg_logger', msg_logging_dir, level=logging.INFO)
warning_logging_dir = os.path.join(cfg['base_dir'], cfg['logging_dir'], 'warning_'+str(model_id)+'.txt') 
setup_logger('warning_logger', warning_logging_dir, level=logging.WARNING)

receptive_field, _ = compute_receptive_field_length(cfg['dilations'], cfg['filter_length'], 1)

coord = tf.train.Coordinator()

with tf.name_scope('create_readers'):
                 
    train_file_list = os.path.join(cfg['data_dir'], cfg['train_file_list']) 
    train_clean_audio_dir = os.path.join(cfg['data_dir'], cfg['train_clean_audio_dir'])
    train_noisy_audio_dir = os.path.join(cfg['data_dir'], cfg['train_noisy_audio_dir'])
    train_label_dir = os.path.join(cfg['data_dir'], cfg['train_label_dir']) 
    train_audio_label_reader = AudioConditionsReader(coord, train_file_list, train_clean_audio_dir, train_noisy_audio_dir, 
                                     train_label_dir, cfg['label_dim'], cfg['audio_ext'], cfg['label_ext'], cfg['sample_rate'], 
                                     cfg['noise_only_percent'], cfg['noise_only_percent_gc'], cfg['regain'], cfg['frame_length'], 
                                     cfg['frame_shift'], lc_context_length=cfg['label_context_length'], gc_ids_mapping=cfg['gc_ids_mapping'],
                                     input_length=None, target_length=cfg['target_length'], receptive_field=receptive_field,
                                     silence_threshold=cfg['silence_threshold'], queue_size=cfg['queue_size'], 
                                     permute_segments=cfg['permute_segments'], lc_enabled=cfg['lc_enabled'], gc_enabled=cfg['gc_enabled'])

    valid_file_list = os.path.join(cfg['data_dir'], cfg['valid_file_list']) 
    valid_clean_audio_dir = os.path.join(cfg['data_dir'], cfg['valid_clean_audio_dir'])
    valid_noisy_audio_dir = os.path.join(cfg['data_dir'], cfg['valid_noisy_audio_dir']) 
    valid_label_dir = os.path.join(cfg['data_dir'], cfg['valid_label_dir'])
    valid_audio_label_reader = AudioConditionsReader(coord, valid_file_list, valid_clean_audio_dir, valid_noisy_audio_dir, 
                                     valid_label_dir, cfg['label_dim'], cfg['audio_ext'], cfg['label_ext'], cfg['sample_rate'], 
                                     cfg['noise_only_percent'], cfg['noise_only_percent_gc'], cfg['regain'], cfg['frame_length'], 
                                     cfg['frame_shift'], lc_context_length=cfg['label_context_length'], gc_ids_mapping=cfg['gc_ids_mapping'],
                                     input_length=None, target_length=cfg['target_length'], receptive_field=receptive_field,
                                     silence_threshold=cfg['silence_threshold'], queue_size=cfg['queue_size'], 
                                     permute_segments=cfg['permute_segments'], lc_enabled=cfg['lc_enabled'], gc_enabled=cfg['gc_enabled'])

# define learning rate decay method 
global_step = tf.Variable(0, trainable=False, name='global_step')
learning_rate = get_learning_rate(cfg['learning_rate_method'], global_step, learning_rate_params)

# define the optimization algorithm
opt_name = cfg['optimization_algorithm'].lower()
optimizer = get_optimizer(opt_name, learning_rate, optim_params)


# Create the network
wavenet = Wavenet(cfg, model_id)

# Train the network
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sess = tf.Session(config=config)
sess = tf.Session()

# Recover the parameters of the model
if cfg['model_id'] is not None:
    print('Restore the parameters of model ' + str(cfg['model_id']))
    restore_variables(sess, cfg)
else:
    print('Train new model') 

# Define the train computation graph
wavenet.define_train_computations(optimizer, train_audio_label_reader, valid_audio_label_reader, global_step)

try:
    wavenet.train(cfg, coord, sess)
except KeyboardInterrupt:  
    print()
finally:
    if not coord.should_stop():
        coord.request_stop() 
    sess.close() 



