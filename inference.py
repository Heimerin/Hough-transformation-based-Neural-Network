import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
import random
import glob
import os


def moving_window_inference(heatmap, threshold=0.5, window_size=5):
    """
    Skanuje wygładzoną mapę z U-Netu i wyciąga współrzędne maksimów.
    Threshold jest ułamkiem (np. > 0.5).
    """
    peaks = []
    offset = window_size // 2
    
    for y in range(offset, heatmap.shape[0] - offset):
        for x in range(offset, heatmap.shape[1] - offset):
            center_val = heatmap[y, x]
            if center_val >= threshold:
                window = heatmap[y - offset : y + offset + 1, x - offset : x + offset + 1]
                if center_val == np.max(window):
                    peaks.append((y, x, center_val))
    return peaks


def extract_patch(accumulator_frame, full_peaks_list, start_y, window_size=216):
    image_patch = accumulator_frame[start_y : start_y + window_size, :]
    patch_peaks = []
    
    for peak in full_peaks_list:
        x_coord, y_coord = peak[0], peak[1]  
        if start_y <= y_coord < (start_y + window_size):
            local_y = y_coord - start_y
            patch_peaks.append((local_y, x_coord))
            
    return image_patch, patch_peaks


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIRECTORY = os.path.join(BASE_DIR, "data1") #nazwa folderu z danymi

    model = load_model('unet_cern_real_data.keras')

    file_list = glob.glob(os.path.join(DATA_DIRECTORY, "*.npz"))
    random_file = random.choice(file_list)
    
    data = np.load(random_file, allow_pickle=True)
    original_values = data["original_values"]
    true_peaks = data["true_peaks"]

    if original_values.ndim == 2:
        original_values = original_values[np.newaxis, ...]
        true_peaks = [true_peaks]

    frame_idx = random.randint(0, original_values.shape[0] - 1)
    max_y = original_values.shape[1] - 216
    start_y = random.randint(0, max_y)

    frame_accumulator = original_values[frame_idx]
    frame_peaks = true_peaks[frame_idx]

    patch_X, true_local_peaks = extract_patch(frame_accumulator, frame_peaks, start_y, window_size=216)

    input_tensor = np.expand_dims(patch_X, axis=(0, -1)).astype(np.float32)

    predicted_tensor = model.predict(input_tensor)
    predicted_heatmap = predicted_tensor[0, :, :, 0]

    found_peaks = moving_window_inference(predicted_heatmap, threshold=0.5, window_size=5)


    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    
    axes[0].imshow(patch_X, origin='lower', cmap='viridis')
    axes[0].set_title(f"Surowe Wejście\nPlik: {os.path.basename(random_file)}")

  
    axes[1].imshow(predicted_heatmap, origin='lower', cmap='magma')
    axes[1].set_title("Wygładzona Mapa- wyjście z U-Netu")

    axes[2].imshow(patch_X, origin='lower', cmap='viridis')
    axes[2].set_title(f"Rekonstrukcja: Znaleziono {len(found_peaks)} | Prawdziwe {len(true_local_peaks)}")

    # Zaznaczamy BIAŁYMI OKRĘGAMI to, co podali fizycy w 'true_peaks'
    for y, x in true_local_peaks:
        axes[2].plot(x, y, 'wo', markersize=8, fillstyle='none', label='Prawda' if 'Prawda' not in axes[2].get_legend_handles_labels()[1] else "")

    # Zaznaczamy CZERWONYMI KRZYŻYKAMI to, co znalazł nasz program
    for y, x, conf in found_peaks:
        axes[2].plot(x, y, 'rx', markersize=10, markeredgewidth=2, label='Predykcja (AI)' if 'Predykcja (AI)' not in axes[2].get_legend_handles_labels()[1] else "")

    axes[2].legend(loc='upper right')
    plt.tight_layout()
    plt.show()