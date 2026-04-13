import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
from tensorflow.keras import layers, models

from MC import generate_event
from hough import (fill_hough_accumulator, Q_PT_BINS, PHI_BINS, Q_PT_MIN, Q_PT_MAX, PHI_MIN, PHI_MAX)

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
                    peaks.append((q_pt_idx, phi_idx, center_val))
    return peaks

#2. metod, U Net NETWORK 

#heatmap do Uneta z uzyciem dwuwymiarowego rozkladu Gaussa
#uzywany jest heatmap by nie szkolic sieci binarnie 
def gen_truth_heatmap(true_params, shape=(Q_PT_BINS, PHI_BINS), sigma=2.0):
    heatmap=np.zeros(shape)

    #siatka wspolrzednych
    x_grid, y_grid = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    for q_pt, phi in true_params:
        #transformacja ciaglych wartosci na dyskretne 
        phi_idx = int((phi - PHI_MIN)/ (PHI_MAX - PHI_MIN) * shape[1])
        q_pt_idx = int((q_pt - Q_PT_MIN)/(Q_PT_MAX - Q_PT_MIN) * shape[0])

        if 0 <= phi_idx< shape[1] and 0 <= q_pt_idx < shape[0]:
            #dwuwymiarowy gauss
            dist_sq = (x_grid - phi_idx)**2 + (y_grid - q_pt_idx)**2
            heatmap += np.exp(-dist_sq / (2*sigma**2))

    return np.clip(heatmap, 0.0, 1.0)


#architektura sieci
def build_unet(input_shape=(Q_PT_BINS, PHI_BINS, 1)):
    inputs=layers.Input(shape=input_shape)
    #enkoder 
    c1 = layers.Conv2D(16, (3,3), activation = 'relu', padding='same')(inputs)
    c1 = layers.Conv2D(16, (3,3), activation = 'relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2,2))(c1)

    c2 = layers.Conv2D(16, (3,3), activation = 'relu', padding='same')(p1)
    c2 = layers.Conv2D(16, (3,3), activation = 'relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2,2))(c2)

    #zwezenie sieci
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(64, (3,3), activation='relu', padding='same')(c3)

    #dekoder
    u4 = layers.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same')(c3)
    u4 = layers.concatenate([u4, c2])
    c4=layers.Conv2D(32, (3,3), activation = 'relu', padding='same')(u4)
    c4=layers.Conv2D(32, (3,3), activation = 'relu', padding='same')(c4)
    u5=layers.Conv2DTranspose(16, (2,2), strides=(2,2), padding='same')(c4)
    u5=layers.concatenate([u5, c1])
    c5=layers.Conv2D(16, (3,3), activation = 'relu', padding='same')(u5)
    c5=layers.Conv2D(16, (3,3), activation = 'relu', padding='same')(c5)
    #zwraca pradwopodobienstwo bycia peakem
    outputs=layers.Conv2D(1, (1,1), activation='sigmoid')(c5)
    model=models.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

#blok testowy 
if __name__ == "__main__":
    hit_package, particle_params = generate_event(true_tracks=5, noise_hits=50)
    hough_hist = fill_hough_accumulator(hit_package)
    found_peaks = moving_window(hough_hist, threshold=5, window_size=5)
    heatmap_label=gen_truth_heatmap(particle_params)
    unet_model=build_unet()
    unet_model.summary()

    fig, axes = plt.subplots(1, 2, figsize=(14,6))
    axes[0].imshow(hough_hist, origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title("Zaszumiony Akumulaotor parametrów (wejscie do sieci)")
    axes[1].imshow(heatmap_label, origin='lower', aspect='auto', cmap='magma')
    axes[1].set_title("heatmap docelowy (prawdziwe parametry)")
    plt.show()  
    