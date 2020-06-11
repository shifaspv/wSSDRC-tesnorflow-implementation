import os
import threading
import logging
import math
import logging

import soundfile as sf
import numpy as np
import scipy.signal
import tensorflow as tf
import lib.util as util
from lib.util import load_wav
from lib.precision import _FLOATX

def wav_to_float(x):
    '''try:
        max_value = np.iinfo(x.dtype).max
        min_value = np.iinfo(x.dtype).min
    except:
        max_value = np.finfo(x.dtype).max
        min_value = np.finfo(x.dtype).min
    print(np.min(x), np.max(x))
    x = x.astype("float64", casting='safe')
    x -= min_value
    x /= ((max_value - min_value) / 2.)
    x -= 1.'''
    x = x.astype(_FLOATX.as_numpy_dtype()) #, casting='safe')  
    return x

def read_wav(filename):
    # Reads in a wav audio file, takes the first channel, converts the signal to float32 representation

    audio_signal, sample_rate = sf.read(filename)

    if audio_signal.ndim > 1:
        audio_signal = audio_signal[:, 0]

    if audio_signal.dtype != _FLOATX.name:
        audio_signal = wav_to_float(audio_signal)

    return audio_signal, sample_rate



def get_subsequence_with_speech_indices(full_sequence, min_length, sample_rate, silence_threshold=0.1):
    signal_magnitude = np.abs(full_sequence)

    chunk_length = max(1, int(sample_rate*0.005)) # 5 milliseconds

    chunks_energies = []
    for i in range(0, len(signal_magnitude), chunk_length):
        chunks_energies.append(np.mean(signal_magnitude[i:i + chunk_length]))

    threshold = np.max(chunks_energies) * silence_threshold

    onset_chunk_i = 0
    for i in range(0, len(chunks_energies)):
        if chunks_energies[i] >= threshold:
            onset_chunk_i = i
            break

    termination_chunk_i = len(chunks_energies)
    for i in range(len(chunks_energies) - 1, 0, -1):
        if chunks_energies[i] >= threshold:
            termination_chunk_i = i
            break

    if (termination_chunk_i - onset_chunk_i)*chunk_length >= min_length: # Then pad, else ignore
        num_pad_chunks = 4
        onset_chunk_i = np.max((0, onset_chunk_i - num_pad_chunks))
        termination_chunk_i = np.min((len(chunks_energies), termination_chunk_i + num_pad_chunks))

    return [onset_chunk_i*chunk_length, (termination_chunk_i+1)*chunk_length]


def extract_subsequence_with_speech(clean_audio, noisy_audio, label, min_length, fs, silence_threshold=0.1):

    indices = get_subsequence_with_speech_indices(clean_audio, min_length, fs, silence_threshold)

    if indices[0] == indices[1]:
        return None, None, None
    elif label is None:
        return clean_audio[indices[0]:indices[1]], noisy_audio[indices[0]:indices[1]], None
    else:    
        return clean_audio[indices[0]:indices[1]], noisy_audio[indices[0]:indices[1]], label[indices[0]:indices[1], :]


def read_filelist(file_list, gc_enabled, ids_mapping=None):
    fid = open(file_list, 'r')
    lines = fid.readlines()
    fid.close()
    
    filenames = []

    if gc_enabled: 
        speaker_ids = [] 
    else:
        speaker_ids = None

    for filename in lines:
        filenames.append( filename.rstrip() )
        if gc_enabled: 
           speaker_name = filename[0:4] 
           if speaker_name in ids_mapping:
               speaker_id = ids_mapping[speaker_name]
           else:
               speaker_id = 0   
 
           speaker_ids.append( speaker_id )

    return filenames, speaker_ids

def splice_frames(labels, context_length):
    
    n_frames, label_dim = labels.shape

    if context_length == 1:
        return labels, label_dim

    new_label_dim = label_dim*context_length

    new_labels = np.zeros((n_frames, new_label_dim), dtype=_FLOATX.as_numpy_dtype())

    jfrom = -(context_length//2)
    jto = context_length//2 + 1 

    for i in range(n_frames):
        k1 = 0
        k2 = label_dim 
        for j in range(jfrom, jto):
            if (i+j >= 0) and (i+j < n_frames):
                new_labels[i, k1:k2] =  labels[i+j, :]  
            k1 = k2
            k2 += label_dim

    return new_labels, new_label_dim

def load_label(filename, label_dir, label_dim, label_ext='.lab', sample_rate=16000, frame_length=0.025, frame_shift=0.005,
               context_length=1):
    '''Reads the phonetic unit labels.'''
    

    label_filename = os.path.join(label_dir, filename.rstrip() + label_ext) 

    with open(label_filename, 'rb') as fid:
        label = np.fromfile(fid, dtype=np.float32, count=-1)
    fid.close()

    n_frames = int(len(label)/label_dim) 
    label = label.reshape((n_frames, label_dim))
    #label = label.reshape((label_dim, n_frames))
    #label = np.transpose(label)
    #plt.plot(label[:, 1])
    #plt.show()  

    label, label_dim = splice_frames(label, context_length)       
  
    samples_per_shift = int(sample_rate*frame_shift)
    samples_per_frame = int(sample_rate*frame_length) 
    samples_per_frame = int(round(float(samples_per_frame)/float(samples_per_shift)))*samples_per_shift

    # Upsample labels. There is 80% overlap between frames. Each frame is 0.025 sec. The frame shift is 0.005 sec
    # ---------- frame_0 
    #   ---------- frame_1
    #     ---------- frame_2
    #       ---------- frame_3
    #         ---------- frame_4
    # The following overlap-and-add procedure may not be appropriate for features 417:425 
    n_label_samples = int((n_frames + frame_length/frame_shift - 1)*samples_per_shift)
    upsampled_label = np.zeros((n_label_samples, label_dim), dtype=np.float32)
    count = np.zeros((n_label_samples, ), dtype=np.float32)

    for j in range(n_frames):
        i = j*samples_per_shift
        count[i:i+samples_per_frame] += 1
        for k in range(samples_per_frame):
            upsampled_label[i+k, :] += label[j, :] 

    for j in range(n_label_samples):
        upsampled_label[j, :] /= count[j]    
 
    return upsampled_label


def load_noisy_audio_label_and_speaker_id(filename, test_noisy_audio_dir, audio_ext, lc_enabled, test_label_dir, label_dim, label_ext, 
                                          sample_rate, frame_length, frame_shift, label_context_length, gc_enabled, gc_ids_mapping):

    noisy_audio_fullpathname = os.path.join(test_noisy_audio_dir, filename + audio_ext)
    noisy_audio = load_wav(noisy_audio_fullpathname, sample_rate)

    n_audio_samples = noisy_audio.shape[0]

    if lc_enabled:
        label = load_label(filename, test_label_dir, label_dim, label_ext, sample_rate, frame_length, frame_shift, label_context_length)

        n_label_samples = label.shape[0]
        if (n_audio_samples > n_label_samples):
            noisy_audio = noisy_audio[0:n_label_samples]
            n_audio_samples = n_label_samples 
        elif (n_audio_samples < n_label_samples):
            label = label[0:n_audio_samples, :] 
            n_label_samples = n_audio_samples
    else:
        label = None

    lc = label

    if gc_enabled: 
        speaker_name = filename[0:4] 
        if speaker_name in gc_ids_mapping:
            speaker_id = gc_ids_mapping[speaker_name]
        else:
            speaker_id = 0
        gc = np.empty((n_audio_samples, ), dtype=np.int32)
        gc.fill(speaker_id) 
    else:
        gc = None
   
    return noisy_audio, lc, gc   


def load_clean_noisy_audio_and_label(filename, clean_audio_dir, noisy_audio_dir, lc_enabled, label_dir, label_dim, audio_ext='.wav', label_ext='.lab', sample_rate=16000, frame_length=0.025, frame_shift=0.005, context_length=1):
    '''Reads an audio file and the corresponding phonetic unit labels.'''
    

    clean_audio_fullpathname = os.path.join(clean_audio_dir, filename.rstrip() + audio_ext)
    clean_audio = load_wav(clean_audio_fullpathname, sample_rate)
        
    n_audio_samples = len(clean_audio) 

    noisy_audio_fullpathname = os.path.join(noisy_audio_dir, filename.rstrip() + audio_ext)
    noisy_audio = load_wav(noisy_audio_fullpathname, sample_rate)

    assert(n_audio_samples == len(noisy_audio))
  
    if lc_enabled:
        upsampled_label = load_label(filename, label_dir, label_dim, label_ext, sample_rate, frame_length, frame_shift, context_length)
        n_label_samples = len(upsampled_label)
 
        if (n_audio_samples > n_label_samples):
            clean_audio = clean_audio[0:n_label_samples]
            noisy_audio = noisy_audio[0:n_label_samples] 
        elif (n_audio_samples < n_label_samples):
            upsampled_label = upsampled_label[0:n_audio_samples, :] 
    else:
        upsampled_label = None   
 
    return clean_audio, noisy_audio, upsampled_label



class AudioConditionsReader(object):
    '''Generic background audio and label reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 coord, 
                 file_list,
                 clean_audio_dir,
                 noisy_audio_dir,
                 label_dir,
                 label_dim,
                 audio_ext,
                 label_ext,
                 sample_rate,
                 noise_only_percent,
                 noise_only_percent_gc,
                 regain, 
                 frame_length,
                 frame_shift,
                 lc_context_length=1,
                 gc_ids_mapping=None,
                 input_length=None,
                 target_length=None,
                 receptive_field=32,
                 silence_threshold=None,
                 queue_size=64, 
                 permute_segments=False,
                 lc_enabled=False,
                 gc_enabled=False):

        self.coord = coord
        self.file_list = file_list
        self.clean_audio_dir = clean_audio_dir
        self.noisy_audio_dir = noisy_audio_dir
        self.label_dir = label_dir   
        self.label_dim = label_dim 
        self.audio_ext = audio_ext
        self.label_ext = label_ext  
        self.sample_rate = sample_rate
        self.noise_only_percent = noise_only_percent
        self.noise_only_percent_gc = noise_only_percent_gc 
        self.regain = regain
        self.frame_length = frame_length
        self.frame_shift = frame_shift 
        self.lc_context_length = lc_context_length 
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold 
        self.permute_segments = permute_segments 
        self.lc_enabled = lc_enabled
        self.gc_enabled = gc_enabled

        if input_length is not None:
            self.input_length = input_length
            self.target_length = input_length - (receptive_field - 1)
        elif target_length is not None:
            self.input_length = target_length + (receptive_field - 1)   
            self.target_length = target_length 
        else:
            self.input_length = None
            self.target_length = None

        self.logger = logging.getLogger("warning_logger")

        self.audio_input_placeholder = tf.placeholder(dtype=_FLOATX, shape=(None, )) 
        self.audio_input_queue = tf.FIFOQueue(queue_size, _FLOATX)
        self.audio_input_enqueue_op = self.audio_input_queue.enqueue(self.audio_input_placeholder)

        self.audio_output1_placeholder = tf.placeholder(dtype=_FLOATX, shape=(None, )) 
        self.audio_output1_queue = tf.FIFOQueue(queue_size, _FLOATX)
        self.audio_output1_enqueue_op = self.audio_output1_queue.enqueue(self.audio_output1_placeholder)

        self.audio_output2_placeholder = tf.placeholder(dtype=_FLOATX, shape=(None, )) 
        self.audio_output2_queue = tf.FIFOQueue(queue_size, _FLOATX)
        self.audio_output2_enqueue_op = self.audio_output2_queue.enqueue(self.audio_output2_placeholder)

        if lc_enabled:  
            self.label_placeholder = tf.placeholder(dtype=_FLOATX, shape=(None, label_dim*lc_context_length)) 
            self.label_queue = tf.FIFOQueue(queue_size, _FLOATX)
            self.label_enqueue_op = self.label_queue.enqueue(self.label_placeholder)
     
        if gc_enabled:
            self.gc_placeholder = tf.placeholder(dtype=tf.int32, shape=()) 
            self.gc_queue = tf.FIFOQueue(queue_size, 'int32')
            self.gc_enqueue_op = self.gc_queue.enqueue(self.gc_placeholder) 
     

        self.filenames, self.speaker_ids = read_filelist(file_list, gc_enabled, gc_ids_mapping)

        self.n_files = len(self.filenames)

        self.indices_list = self.find_segment_indices()

        self.n_segments = len(self.indices_list) 
    
        self.perm_indices = np.arange(self.n_segments) 

        self.reset()

    def reset(self):
        self.enqueue_finished = False
        self.dequeue_finished = False
        self.n_enqueued = 0
        self.n_dequeued = 0

        if self.permute_segments: 
            np.random.shuffle(self.perm_indices) 

    
    def find_segment_indices(self):

        indices_list = []
        
        for i, filename in enumerate(self.filenames):
            clean_audio_fullpathname = os.path.join(self.clean_audio_dir, filename.rstrip() + self.audio_ext)
            audio = load_wav(clean_audio_fullpathname, self.sample_rate)

            if self.silence_threshold > 0:
                # Remove silence 
                indices = get_subsequence_with_speech_indices(audio, self.receptive_field, self.sample_rate, self.silence_threshold)
                if indices[0] == indices[1]:
                    audio = None
                else: 
                    audio = audio[indices[0]:indices[1]]

            if (audio is None):
                self.logger.warning("Warning: {} was ignored as it contains only "
                      "silence. Consider decreasing the silence threshold.".format(filename)) 
                continue 

            regain_factor = self.regain / util.rms(audio)

            if self.gc_enabled:
                speaker_id = self.speaker_ids[i]
            else:
                speaker_id = None

            # Cut samples into pieces of size (receptive_field - 1)/2 + target_length with (receptive_field - 1)/2 overlap
            n_audio_samples = len(audio)

            if n_audio_samples < self.receptive_field:
                continue

            if self.input_length is None:
                self.input_length = n_audio_samples
                self.target_length = n_audio_samples - (self.receptive_field - 1)  
                from_index = 0
                to_index = n_audio_samples 
                indices_list.append((filename, from_index, to_index, regain_factor, speaker_id)) 
            else:
                from_index = 0
                to_index = self.input_length
                while n_audio_samples - from_index >= self.receptive_field + int(0.1*self.target_length):
                    if to_index > n_audio_samples:
                        from_index = max(0, n_audio_samples - self.input_length)
                        to_index = n_audio_samples 

                    indices_list.append((filename, from_index, to_index, regain_factor, speaker_id))
                    from_index += self.target_length
                    to_index += self.target_length

        return indices_list
      

    def check_for_elements_and_increment(self): 
   
        if self.enqueue_finished and (self.n_enqueued == self.n_dequeued):
            return False
        else:
            self.n_dequeued += 1 
            return True


    def dequeue(self): 
        audio_input = self.audio_input_queue.dequeue()
        audio_output1 = self.audio_output1_queue.dequeue()
        audio_output2 = self.audio_output2_queue.dequeue()

        audio_shape = tf.shape(audio_input) 
        n_samples = audio_shape[0]   

        if self.lc_enabled: 
            label = self.label_queue.dequeue() 
        else:
            label = None   

        if self.gc_enabled:
            speaker_id = self.gc_queue.dequeue()
            speaker_id_upsampled = tf.fill((n_samples, ), speaker_id) 
        else:
            speaker_id_upsampled = None 

        return audio_input, audio_output1, label, speaker_id_upsampled, n_samples


    def enqueue_thread(self, sess):
      
        for i in self.perm_indices:
            filename, from_index, to_index, regain_factor, speaker_id = self.indices_list[i]

            clean_audio, noisy_audio, label = load_clean_noisy_audio_and_label(filename, self.clean_audio_dir, self.noisy_audio_dir, 
                               self.lc_enabled, self.label_dir, self.label_dim, self.audio_ext, self.label_ext, self.sample_rate,
                               self.frame_length, self.frame_shift, self.lc_context_length)  

            if self.silence_threshold:
                # Remove silence from the beginning and end of a utterance
                clean_audio, noisy_audio, label = extract_subsequence_with_speech(clean_audio, noisy_audio, label,
                                                               self.receptive_field, self.sample_rate, self.silence_threshold)  

            noise = noisy_audio
            from_index=int(from_index)
            to_index=int(to_index)
            clean_audio_segment = clean_audio[from_index:to_index]
            noise_segment = noise[from_index:to_index]

            clean_audio_segment_regained = clean_audio_segment * regain_factor
            noise_segment_regained = noise_segment * regain_factor

            input_segment =  noise_segment_regained

            output1_segment = clean_audio_segment_regained
            output2_segment = noise_segment_regained

            if label is not None: 
                label_segment = label[from_index:to_index, :]  

            if self.noise_only_percent > 0:
                if np.random.uniform(0, 1) <= self.noise_only_percent:
                    input_segment = noise_segment_regained
                    output1_segment = np.array([0]*input_segment.shape[0], dtype=_FLOATX.as_numpy_dtype()) 

            if self.noise_only_percent_gc > 0: 
                if np.random.uniform(0,1) <= self.noise_only_percent_gc:
                    speaker_id = 0
            
# modification to read single output
            if self.lc_enabled and self.gc_enabled:
                sess.run([self.audio_input_enqueue_op, self.audio_output1_enqueue_op, 
                          self.label_enqueue_op, self.gc_enqueue_op], 
                          feed_dict={self.audio_input_placeholder: input_segment, self.audio_output1_placeholder: output1_segment,
                                      self.label_placeholder: label_segment,
                                     self.gc_placeholder: speaker_id})
            elif self.lc_enabled and (not self.gc_enabled):  
                sess.run([self.audio_input_enqueue_op, self.audio_output1_enqueue_op, self.label_enqueue_op], 
                          feed_dict={self.audio_input_placeholder: input_segment, self.audio_output1_placeholder: output1_segment,
                                      self.label_placeholder: label_segment})
            elif (not self.lc_enabled) and self.gc_enabled:
                sess.run([self.audio_input_enqueue_op, self.audio_output1_enqueue_op, self.gc_enqueue_op], 
                          feed_dict={self.audio_input_placeholder: input_segment, self.audio_output1_placeholder: output1_segment,
                                      self.gc_placeholder: speaker_id}) 
            else: # (not self.lc_enabled) and (not self.gc_enabled):
                sess.run([self.audio_input_enqueue_op, self.audio_output1_enqueue_op], 
                          feed_dict={self.audio_input_placeholder: input_segment, self.audio_output1_placeholder: output1_segment
                                     })
                          
            self.n_enqueued += 1
                       
        self.enqueue_finished = True
                    
                    
    def start_enqueue_thread(self, sess):
        thread = threading.Thread(target=self.enqueue_thread, args=(sess, ))
        thread.start()
        return thread
