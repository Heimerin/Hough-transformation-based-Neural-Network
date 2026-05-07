import numpy as np
import matplotlib.pyplot as plt

def load_and_inspect_npz(filename, frame_idx=0):
    data = np.load(filename, allow_pickle=True)
    original_values = data["original_values"]
    true_peaks = data["true_peaks"]

    if original_values.ndim == 2:
        original_values = original_values[np.newaxis, ...]
    print(f"wymiary aku {original_values.shape}")
    print(f"Ilość wszystkich maksimów/ramka {frame_idx}: {len(true_peaks[frame_idx])}")

    return original_values, true_peaks


def extract_patch(accumulator_frame, full_peaks_list, start_y, window_size=216):

    image_patch = accumulator_frame[start_y : start_y + window_size, :]
    patch_peaks = []

    for peak in full_peaks_list:
        y_coord, x_coord = peak[0], peak[1]

        if start_y <= y_coord < (start_y + window_size):
            local_y = y_coord - start_y
            patch_peaks.append((local_y, x_coord))
    
    return image_patch, patch_peaks

#test
if __name__ == "__main__":
    FILENAME = "hough_0.npz"

    try:
        TARGET_FRAME = 0
        original_values, true_peaks = load_and_inspect_npz(FILENAME, frame_idx=TARGET_FRAME)
        START_CUT = 2000
        frame_accumulator = original_values[TARGET_FRAME]
        frame_peaks = true_peaks[TARGET_FRAME]

        image_patch, filtered_peaks = extract_patch(frame_accumulator, frame_peaks, START_CUT)

        for y, x in filtered_peaks:
            print(f"Y: {y:.1f}, X: {x:.1f}")
        plt.imshow(image_patch, origin='lower', cmap='viridis')
        plt.show()
    except FileNotFoundError:
        print(f"Nie można znaleźć pliku: {FILENAME}")