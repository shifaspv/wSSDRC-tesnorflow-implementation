from __future__ import division, print_function

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.

import os
import sys
import time
from datetime import datetime
import logging
import math
import numpy as np
import tensorflow as tf
from lib.precision import _FLOATX
from lib.ops import conv_via_matmul, concat_relu
from lib.model_io import save_variables, get_info
from lib.util import compute_receptive_field_length, l1_l2_loss
import lib.util as util
import pdb

def get_var_maybe_avg(var_name, ema, **kwargs):
    ''' utility for retrieving polyak averaged params '''
    v = tf.get_variable(var_name, **kwargs)
    if ema is not None:
        v = ema.average(v)
    return v

def get_weight_variable(name, shape=None, initializer=tf.contrib.layers.xavier_initializer_conv2d(), ema=None):
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)

def get_bias_variable(name, shape=None, initializer=tf.constant_initializer(value=0.0, dtype=_FLOATX), ema=None): 
    if shape is None:
        return get_var_maybe_avg(name, ema)
    else:  
        return get_var_maybe_avg(name, ema, shape=shape, dtype=_FLOATX, initializer=initializer)
   


class Wavenet(object):

    def __init__(self, cfg, model_id=None):
        self.cfg = cfg
        self.n_residual_channels = cfg['n_residual_channels']   
        self.n_skip_channels = cfg['n_skip_channels'] 
        self.n_post1_channels = cfg['n_post1_channels']
        self.n_post2_channels = cfg['n_post2_channels'] 
        self.filter_length = cfg['filter_length'] 
        assert(self.filter_length == 3), 'This version works with filter_length = 3'
        self.dilations = cfg['dilations']
        self.use_biases = cfg['use_biases']
        self.l2 = cfg['L2_regularization'] 
        self.lc_enabled = cfg['lc_enabled'] 
        self.gc_enabled = cfg['gc_enabled'] 
        self.gc_cardinality = cfg['gc_cardinality'] 
        self.condition_postprocessing = cfg['condition_postprocessing']
        self.label_dim = cfg['label_dim']*cfg['label_context_length']
        self.use_ema = cfg['use_ema']
        self.model_id = model_id    
        self.receptive_field, self.receptive_field_of_blocks = compute_receptive_field_length(self.dilations, self.filter_length, 1)
        self.half_receptive_field = int(self.receptive_field//2) 

        self.create_variables()

        if self.use_ema:
            self.ema = tf.train.ExponentialMovingAverage(decay=cfg['polyak_decay'])
            trainable_variables = tf.trainable_variables() 
            self.maintain_averages_op = tf.group(self.ema.apply(trainable_variables))
        else:
            self.ema = None
        

    def create_variables(self):
        r = self.n_residual_channels
        s = self.n_skip_channels
        p1 = self.n_post1_channels
        p2 = self.n_post2_channels
        fl = self.filter_length
        n_blocks = len(self.dilations)
         
        with tf.variable_scope('wavenet'):
            with tf.name_scope('conditioning'):
                if self.lc_enabled:
                    l = self.label_dim
                    get_weight_variable('W_lc', (l, 2*r))
                if self.gc_enabled:
                    g = self.gc_cardinality 
                    get_weight_variable('W_gc', (g, 2*r), initializer=tf.random_uniform_initializer(minval=-1, maxval=1))  
               

            with tf.name_scope('causal_layer'):
                get_weight_variable('W', (fl, r))  # q -> 1     
                if self.use_biases['causal_layer']:
                    get_bias_variable('b', (r))

            # Residual blocks  
            with tf.variable_scope('residual_blocks'):
                for i, dilation in enumerate(self.dilations):        
                    with tf.variable_scope('block{}'.format(i)):
                        get_weight_variable('filter_gate_W', (fl*r, 2*r)) 
                        if self.use_biases['filter_gate']:
                            get_bias_variable('filter_gate_b', (2*r)) 

                        if self.lc_enabled or self.gc_enabled:
                            get_weight_variable('c_filter_gate_W', (2*r, 2*r))       

                        # module's output signal 
                        get_weight_variable('output_W', (r, r))
                        if self.use_biases['output']:
                            get_bias_variable('output_b', (r))    
                    
                   
            with tf.name_scope('postprocessing'):
                # Perform concatenation -> 1x1 conv -> Concat_ReLU -> 3x1 conv -> Concat_ReLU -> 3x1 conv
                # Note that it is faster to perform concatenation of skip outputs and then a single matrix multiplication with weights than
                # a matrix multiplication of each skip output with the corresponding weights and then summations of the results.
                # skip connections
                get_weight_variable('skip_W', (n_blocks*r, p1))  
                if self.use_biases['skip']:
                    get_bias_variable('skip_b', (p1))

                get_weight_variable('postprocessing1_W', (fl*2*p1, p2)) # 2*s due to concat_ReLU
                if self.use_biases['postprocessing1']:
                    get_bias_variable('postprocessing1_b', (p2))  

                get_weight_variable('postprocessing1_c_filter_gate_W', (2*r, p2)) 

                get_weight_variable('postprocessing2_W', (fl*2*p2, p2)) # 2*s due to concat_ReLU
                if self.use_biases['postprocessing2']:
                    get_bias_variable('postprocessing2_b', (p2))  

                get_weight_variable('postprocessing2_c_filter_gate_W', (2*r, p2))

                get_weight_variable('postprocessing_project_W', (p2, 1))
                 
                

    def causal_layer(self, X, ema=None):
        # X is a one-dimensional numpy array with values in -1.0..1.0
        X = tf.reshape(X, (-1, 1))
        with tf.name_scope('causal_layer'):
            W = get_weight_variable('W', ema=ema)     
            Y = conv_via_matmul(X, W)
            if self.use_biases['causal_layer']:
                b = get_bias_variable('b', ema=ema) 
                Y += b
 
            Y = tf.tanh(Y) 

        return Y

    def residual_block(self, block_input, condition, index, r, dilation, output_width=None, ema=None, is_last_block=False):
        with tf.variable_scope('residual_blocks'):
            with tf.variable_scope('block{}'.format(index)):
                W = get_weight_variable('filter_gate_W', ema=ema) 
                Y = conv_via_matmul(block_input, W, dilation)

                if self.use_biases['filter_gate']:
                    b = get_bias_variable('filter_gate_b', ema=ema) 
                    Y += b

                if self.lc_enabled or self.gc_enabled:
                    W_c = get_weight_variable('c_filter_gate_W', ema=ema)
                    Y += tf.matmul(condition, W_c) # add conditioning
                
                Y = tf.tanh(Y[:, :r])*tf.sigmoid(Y[:, r:])

                # skip signal
                skip_cut = (tf.shape(Y)[0] - output_width)//2
                if is_last_block:
                    skip_output = Y     
                else:
                    skip_output = Y[skip_cut:-skip_cut, :]  
                     

                # module's output signal 
                W = get_weight_variable('output_W', ema=ema)
                Z = tf.matmul(Y, W) # 1x1 convolution
                if self.use_biases['output']:
                    b = get_bias_variable('output_b', ema=ema)     
                    Z += b

                block_output = block_input[dilation:-dilation, :] + Z  # reconstruct output from input and residual signals.

        return block_output, skip_output

    def postprocessing(self, skip_outputs, condition, ema=None):
        with tf.name_scope('postprocessing'):
            # Perform concatenation -> 1x1 conv -> Concat_ReLU -> 3x1 conv -> Concat_ReLU -> 3x1 conv
            X_concat = tf.concat(skip_outputs, axis=1)
            W = get_weight_variable('skip_W', ema=ema)
            X = tf.matmul(X_concat, W) # 1x1 convolution
            if self.use_biases['skip']:
                b = get_bias_variable('skip_b', ema=ema)
                X += b

            X = concat_relu(X)

            W = get_weight_variable('postprocessing1_W', ema=ema) 
            X = conv_via_matmul(X, W) # 3x1 convolution
            if self.use_biases['postprocessing1']:
                b = get_bias_variable('postprocessing1_b', ema=ema)  
                X += b

            if self.condition_postprocessing and (self.lc_enabled or self.gc_enabled):
                W_c = get_weight_variable('postprocessing1_c_filter_gate_W', ema=ema)
                X += tf.matmul(condition[1:-1, :], W_c) # add conditioning

            X = concat_relu(X) 

            W = get_weight_variable('postprocessing2_W', ema=ema) 
            X = conv_via_matmul(X, W) # 3x1 convolution
            if self.use_biases['postprocessing2']:
                b = get_bias_variable('postprocessing2_b', ema=ema)  
                X += b

            if self.condition_postprocessing and (self.lc_enabled or self.gc_enabled):
                W_c = get_weight_variable('postprocessing2_c_filter_gate_W', ema=ema)
                X += tf.matmul(condition[2:-2], W_c) # add conditioning

            project_W = get_weight_variable('postprocessing_project_W', ema=ema)
            X = tf.matmul(X, project_W) # 1x1 convolution
            X = tf.squeeze(X)

        return X
        
    def project_conditions(self, lc=None, gc=None, ema=None):
        with tf.name_scope('conditioning'):
            if self.lc_enabled: 
                W_lc = get_weight_variable('W_lc', ema=ema)
                H_lc = tf.matmul(lc, W_lc) 
            if self.gc_enabled:
                W_gc = get_weight_variable('W_gc', ema=ema)
                H_gc = tf.nn.embedding_lookup(W_gc, gc)
            if self.lc_enabled and self.gc_enabled:
                H = H_lc + H_gc 
            elif self.lc_enabled:
                H = H_lc
            else:
                H = H_gc

        return tf.nn.tanh(H)

    def get_out_1_loss(self, Y_true, Y_pred):

        weight = self.cfg['loss']['out_1']['weight']
        l1_weight = self.cfg['loss']['out_1']['l1']
        l2_weight = self.cfg['loss']['out_1']['l2']


        if weight == 0:
            return 0

        return weight * l1_l2_loss(Y_true, Y_pred, l1_weight, l2_weight)

    def get_out_2_loss(self, Y_true, Y_pred):

        weight = self.cfg['loss']['out_2']['weight']
        l1_weight = self.cfg['loss']['out_2']['l1']
        l2_weight = self.cfg['loss']['out_2']['l2'] 

        if weight == 0:
            return 0

        return weight * l1_l2_loss(Y_true, Y_pred, l1_weight, l2_weight)
 

    def inference(self, X, lc, gc, ema): 
        # Input X is mixed signal (clean_speech + noise)

        r = self.n_residual_channels  

        input_width  = tf.shape(X)[0]
        output_width = input_width - int(self.receptive_field_of_blocks) + 1

        with tf.variable_scope('wavenet', reuse=True):
            #X -> causal_layer -> residual_block0 -> ... residual_blockl -> postprocessing -> P params
 
            if self.lc_enabled or self.gc_enabled:
                H = self.project_conditions(lc, gc, ema)

            # causal layer
            X = self.causal_layer(X, ema) 
            j = 1

            # Residual blocks  
            last_block_index = len(self.dilations) - 1
            H1 = None
            skip_outputs = []
            for i, dilation in enumerate(self.dilations):
                j = j + dilation
                if self.lc_enabled or self.gc_enabled:
                    H1 = H[j:-j, :] 
                X, skip_output = self.residual_block(X, H1, i, r, dilation, output_width, ema, i==last_block_index)
                skip_outputs.append(skip_output)  

            # Post-processing
            clean_speech_pred = self.postprocessing(skip_outputs, H1, ema)

        return clean_speech_pred
                    
                
                       
    
    def define_train_computations(self, optimizer, train_audio_conditions_reader, valid_audio_conditions_reader, global_step):
        # Train operations 
        self.train_audio_conditions_reader = train_audio_conditions_reader

        mixed_audio_train, clean_audio_train, lc_train, gc_train, self.n_train_samples = train_audio_conditions_reader.dequeue()
  
        clean_audio_train = clean_audio_train[self.half_receptive_field:-self.half_receptive_field]  # target1
        noisy_audio_train = mixed_audio_train[self.half_receptive_field:-self.half_receptive_field]

        clean_audio_pred = self.inference(mixed_audio_train, lc_train, gc_train, ema=None)
        # predicted_noise = mixed_speech - estimated_clean_speech
#        noise_pred = mixed_audio_train[self.half_receptive_field:-self.half_receptive_field] - clean_audio_pred 

        # Loss of train data
        self.train_loss = self.get_out_1_loss(clean_audio_train, clean_audio_pred) #+ self.get_out_1_loss(noisy_audio_train, clean_audio_pred) 

        # Regularization loss 
        if self.l2 is not None:
            l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if not('_b' in v.name)])
            self.train_loss += self.l2*l2_loss

        trainable_variables = tf.trainable_variables()
        self.gradients_update_op = optimizer.minimize(self.train_loss, global_step=global_step, var_list=trainable_variables)
        if self.use_ema:
            self.update_op = tf.group(self.gradients_update_op, self.maintain_averages_op)
        else:
            self.update_op = self.gradients_update_op

        # Validation operations
        self.valid_audio_conditions_reader = valid_audio_conditions_reader

        mixed_audio_valid, clean_audio_valid, lc_valid, gc_valid, self.n_valid_samples = valid_audio_conditions_reader.dequeue()

        clean_audio_valid = clean_audio_valid[self.half_receptive_field:-self.half_receptive_field]  # target 1
        noisy_audio_valid = mixed_audio_valid[self.half_receptive_field:-self.half_receptive_field]  # target 2 

        clean_audio_pred_valid = self.inference(mixed_audio_valid, lc_valid, gc_valid, ema=self.ema)
#        noise_pred_valid = mixed_audio_valid[self.half_receptive_field:-self.half_receptive_field] - clean_audio_pred_valid 

        # Loss of validation data
        self.valid_loss = self.get_out_1_loss(clean_audio_valid, clean_audio_pred_valid) #+ self.get_out_1_loss(noisy_audio_valid, clean_audio_pred_valid)


    def train_epoch(self, coord, sess, logger):
        self.train_audio_conditions_reader.reset()
        thread = self.train_audio_conditions_reader.start_enqueue_thread(sess) 

        train_loss = 0
        total_samples = 0 
        
        while (not coord.should_stop()) and self.train_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss, n_samples, _ = sess.run([self.train_loss, self.n_train_samples, self.update_op]) 
            if math.isnan(batch_loss):
                logger.critical('train cost is NaN')
                coord.request_stop() 
                break 
            train_loss += batch_loss
            total_samples += n_samples  
        
        coord.join([thread])
        
        if total_samples > 0:  
            train_loss /= total_samples  

        return train_loss 
        

    def valid_epoch(self, coord, sess, logger):
        self.valid_audio_conditions_reader.reset()
        thread = self.valid_audio_conditions_reader.start_enqueue_thread(sess) 

        valid_loss = 0
        total_samples = 0 

        while (not coord.should_stop()) and self.valid_audio_conditions_reader.check_for_elements_and_increment():
            batch_loss, n_samples = sess.run([self.valid_loss, self.n_valid_samples])
            if math.isnan(batch_loss):
                logger.critical('valid cost is NaN')
                coord.request_stop()
                break  
            valid_loss += batch_loss
            total_samples += n_samples  

        coord.join([thread])  

        if total_samples > 0:  
            valid_loss /= total_samples  

        return valid_loss

    def train(self, cfg, coord, sess):
        logger = logging.getLogger("msg_logger") 

        started_datestring = "{0:%Y-%m-%d, %H-%M-%S}".format(datetime.now())
        logger.info('Training of WaveNet started at: ' + started_datestring + ' using Tensorflow.\n')
        logger.info(get_info(cfg))

        start_time = time.clock()

        n_early_stop_epochs = cfg['n_early_stop_epochs']
        n_epochs = cfg['n_epochs']

        saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=4)

        early_stop_counter = 0

        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        sess.run(init_op) 
     
        min_valid_loss = sys.float_info.max
        epoch = 0
        while (not coord.should_stop()) and (epoch < n_epochs):
            epoch += 1
            epoch_start_time = time.clock() 
            train_loss = self.train_epoch(coord, sess, logger) 
            valid_loss = self.valid_epoch(coord, sess, logger) 

            epoch_end_time = time.clock()
                         
            info_str = 'Epoch=' + str(epoch) + ', Train: ' + str(train_loss) + ', Valid: '
            info_str += str(valid_loss) + ', Time=' + str(epoch_end_time - epoch_start_time)  
            logger.info(info_str)

            if valid_loss < min_valid_loss: 
                logger.info('Best epoch=' + str(epoch)) 
                save_variables(sess, saver, epoch, cfg, self.model_id) 
                min_valid_loss = valid_loss 
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            if early_stop_counter > n_early_stop_epochs:
                # too many consecutive epochs without surpassing the best model
                logger.debug('stopping early')
                break

        end_time = time.clock()
        logger.info('Total time = ' + str(end_time - start_time))

        if (not coord.should_stop()):
            coord.request_stop()

    def generation(self, sess, noisy_audio, lc=None, gc=None):
        n_samples = noisy_audio.shape[0]  
#        pdb.set_trace()
        noisy_audio = (0.06 / util.rms(noisy_audio))* noisy_audio
        noisy_audio=np.append(np.zeros((self.half_receptive_field,), dtype=np.float32), noisy_audio)
        noisy_audio=np.append(noisy_audio,np.zeros((self.half_receptive_field,),dtype=np.float32))
#        noisy_audio.astype('float32') 
        
        clean_audio_pred = self.inference(noisy_audio, lc, gc, ema=self.ema)
        noise_pred = noisy_audio[self.half_receptive_field:-self.half_receptive_field] - clean_audio_pred 

        clean_audio_pred_np, noise_pred_np = sess.run([clean_audio_pred, noise_pred])

        return clean_audio_pred_np, noise_pred_np
