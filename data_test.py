import numpy as np
import matplotlib.pyplot as plt

def load_and_inspect_npz(filename, frame_idx=0):
    data = np.load(filename, allow_pickle=True)
    original_values = data["original_values"]
    true_peaks = data["true_peaks"]
    
    if original_values.ndim == 2:
        original_values = original_values[np.newaxis, ...]
        
    return original_values, true_peaks

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


if __name__ == "__main__":
    FILENAME = "hough_1.npz" 
    
    try:
        original_values, true_peaks = load_and_inspect_npz(FILENAME, frame_idx=0)
        frame_accumulator = original_values[0]
        frame_peaks = true_peaks[0]
        
        START_CUT = 2000
        image_patch, filtered_peaks = extract_patch(frame_accumulator, frame_peaks, START_CUT)
        
        patch_heatmap = generate_patch_heatmap(image_patch.shape, filtered_peaks, sigma=2.5)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        axes[0].imshow(image_patch, origin='lower', cmap='viridis')
        axes[0].set_title(f"Surowy fragment (Start Y={START_CUT})")
        axes[0].set_xlabel("Oś X (q/p_T)")
        axes[0].set_ylabel("Oś Y")
        
        axes[1].imshow(patch_heatmap, origin='lower', cmap='magma')
        axes[1].set_title(f"Mapa cieplna")
        axes[1].set_xlabel("Oś X (q/p_T)")

        for y, x in filtered_peaks:
            axes[1].plot(x, y, 'wx', markersize=10) 
            
        plt.tight_layout()
        plt.show()
        
    except FileNotFoundError:
        print(f"BŁĄD {FILENAME}.")