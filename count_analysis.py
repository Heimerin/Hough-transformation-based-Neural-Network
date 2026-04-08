import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tenssorflow import layers, models

from MC import generate_event
from hough import (fill_hough_generator, Q_PT_BINS, PHI_BINS, Q_PT_MIN, Q_PT_MAX, PHI_MIN, PHI_MAX)

#1 metod, moving window 
def moving_window(accumulator, threshold=6, window_size=5):
    peaks=[]
    offset = window_size // 2

    for q_pt_idx in range(offset, Q_PT_BINS - offset):
        for phi_idx in range(offset, PHI_BINS - offset):
            center_val = accumulator[q_pt_idx, phi_idx]
            if center_val >= threshold:
                window = accumulator[q_pt_idx  - offset: q_pt_idx + offset+1, phi_idx - offset: phi_idx + offset + 1]
                if center_val == np.max(window):
                    peaks.ppend((q_pt_idx, phi_idx, center_val))
    return peaks

#2. metod, U Net NETWORK 

