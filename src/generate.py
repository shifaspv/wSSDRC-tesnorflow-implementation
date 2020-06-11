from __future__ import print_function

__author__ = """Vassilis Tsiaras (tsiaras@csd.uoc.gr)"""
#    Vassilis Tsiaras <tsiaras@csd.uoc.gr>
#    Computer Science Department, University of Crete.

import os
import logging
import tensorflow as tf
from model import Wavenet
from lib.optimizers import get_learning_rate, get_optimizer
from lib.audio_conditions_io import AudioConditionsReader, load_noisy_audio_label_and_speaker_id
from lib.model_io import get_configuration, setup_logger
from lib.model_io import restore_variables
from lib.util import load_wav, write_wav

cfg, _, _, speaker_id = get_configuration('generation')

os.environ["CUDA_VISIBLE_DEVICES"] = cfg["CUDA_VISIBLE_DEVICES"]

coord = tf.train.Coordinator()
sess = tf.Session()

# Create the network
wavenet = Wavenet(cfg)

if cfg['use_ema']:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

# Recover the parameters of the model
restore_variables(sess, cfg)


# Generate waveform
noisy_audio, lc, gc = load_noisy_audio_label_and_speaker_id(cfg['noisy_speech_filename'], cfg['test_noisy_audio_dir'], cfg['audio_ext'], 
                                                            cfg['lc_enabled'], cfg['test_label_dir'], cfg['label_dim'], cfg['label_ext'], 
                                                            cfg['sample_rate'], cfg['frame_length'], cfg['frame_shift'], 
                                                            cfg['label_context_length'], cfg['gc_enabled'], cfg['gc_ids_mapping'])


clean_audio, noise = wavenet.generation(sess, noisy_audio, lc, gc)   

wav_out_path = os.path.join(cfg['base_dir'], cfg['output_dir'], str(cfg['model_id']))
if not os.path.exists(wav_out_path):
    os.makedirs(wav_out_path)

clean_audio_out_fullpathname = os.path.join(wav_out_path, cfg['output_clean_speech_filename'] + cfg['audio_ext']) 
print(clean_audio_out_fullpathname, clean_audio.shape, clean_audio.dtype, cfg['sample_rate']) 
write_wav(clean_audio, clean_audio_out_fullpathname, cfg['sample_rate'])

#noise_out_fullpathname = os.path.join(wav_out_path, cfg['output_noise_filename'] + cfg['audio_ext']) 
#write_wav(noise, noise_out_fullpathname, cfg['sample_rate'])
