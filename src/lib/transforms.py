from __future__ import division

import numpy as np
import tensorflow as tf
from lib.precision import _FLOATX


def one_hot_encode(X, n_classes):
    n_samples = np.shape(X)[0]
    Y = np.zeros((n_samples, n_classes), dtype=_FLOATX.as_numpy_dtype())
    Y[np.arange(n_samples), X] = 1
    return Y

def mu_law(audio, mu=255.0):
    # Perform mu-law companding transformation (ITU-T, 1988).
    # Minimum operation is here to deal with rare large amplitudes caused
    # by resampling.
    safe_audio_abs = np.minimum(np.abs(audio), 1.0)
    magnitude = np.log(1.0 + mu * safe_audio_abs) / np.log(1.0 + mu)
    companded_audio = np.sign(audio) * magnitude
    
    return companded_audio.astype(_FLOATX.as_numpy_dtype())



def mu_law_inverse(companded_audio, mu=255.0):
    # Perform inverse of mu-law transformation.
    magnitude = (1.0 / mu) * ((1 + mu)**abs(companded_audio) - 1)
    audio = np.sign(companded_audio) * magnitude

    return audio.astype(_FLOATX.as_numpy_dtype())




