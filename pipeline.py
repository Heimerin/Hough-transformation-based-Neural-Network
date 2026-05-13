import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import glob
import random
import os 
from count_analysis import build_unet

def extract_patch(accumulator_frame, full_peaks_list, start_y, window_size=216):
    image_patch = accumulator_frame[start_y : start_y + window_size, :]
    patch_peaks = []
    
    for peak in full_peaks_list:
        x_coord, y_coord = peak[0], peak[1]  
        
        if start_y <= y_coord < (start_y + window_size):
            local_y = y_coord - start_y
            patch_peaks.append((local_y, x_coord))
            
    return image_patch, patch_peaks

def generate_patch_heatmap(patch_shape, patch_peaks, sigma=2.0):
    heatmap = np.zeros(patch_shape)
    y_grid, x_grid = np.mgrid[0:patch_shape[0], 0:patch_shape[1]]
    
    for peak_y, peak_x in patch_peaks:
        dist_sq = (x_grid - peak_x)**2 + (y_grid - peak_y)**2
        heatmap += np.exp(-dist_sq / (2 * sigma**2))
        
    return np.clip(heatmap, 0.0, 1.0)

def data_generator(data_dir, patches_per_file=64, patch_size=216):
    file_list = glob.glob(os.path.join(data_dir, "*.npz"))
    while True:
        random_file = random.choice(file_list)
        data = np.load(random_file, allow_pickle=True)
        original_values = data["original_values"]
        true_peaks=data["true_peaks"]

        if original_values.ndim == 2:
            original_values = original_values[np.newaxis, ...]
            true_peaks = [true_peaks]
        
        num_frames = original_values.shape[0]
        max_y = original_values.shape[1] - patch_size

        for i in range(patches_per_file):
            frame_idx = random.randint(0, num_frames -1)
            start_y = random.randint(0, max_y)

            frame_accumulator = original_values[frame_idx]
            frame_peaks = true_peaks[frame_idx]

            patch_X, local_peaks = extract_patch(frame_accumulator, frame_peaks, start_y, patch_size)
            patch_Y = generate_patch_heatmap(patch_X.shape, local_peaks, sigma=3.0)

            #dostosowanie wymiaru pod U net
            final_X = np.expand_dims(patch_X, axis=-1)
            final_Y = np.expand_dims(patch_Y, axis = -1)
            yield final_X, final_Y

#test
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "data1")    #nazwa folderu z danymi

    BATCH_SIZE = 8
    PATCH_SIZE = 216
    STEPS_PER_EPOCH = 250
    EPOCHS = 10

    output_sig = (tf.TensorSpec(shape=(PATCH_SIZE, PATCH_SIZE, 1), dtype=tf.float32),tf.TensorSpec(shape=((PATCH_SIZE, PATCH_SIZE, 1)), dtype=tf.float32))
    
    dataset = tf.data.Dataset.from_generator(
        lambda: data_generator(DATA_DIRECTORY, patches_per_file=64, patch_size=PATCH_SIZE),
        output_signature=output_sig
    ).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    model = build_unet(input_shape=(PATCH_SIZE, PATCH_SIZE, 1))

    checkpoint = ModelCheckpoint(
        filepath='unet_cern_real_data.keras', 
        monitor='loss',               
        save_best_only=True,          
        verbose=1                     
    )

    history = model.fit(
        dataset,
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        callbacks=[checkpoint]
    )