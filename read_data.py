import numpy as np
import matplotlib.pyplot as plt

def display_image_grid(filename, cols=4):
    """Display image stack in a grid"""
    
    # Added allow_pickle=True to handle true_peaks saved as an object array (to handle inhomogeneous shapes)
    data = np.load(filename, allow_pickle=True)  
    values = data["values"]
    values_out = data["original_values"]
    
    # Handle dimensions
    if values.ndim == 2:
        values = values[np.newaxis, ...]
        values_out = values_out[np.newaxis, ...]
    
    n_frames = values.shape[0]
    rows = (n_frames + cols - 1) // cols
    
    # Create figure for values
    fig1, axes1 = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes1 = axes1.flatten() if rows*cols > 1 else [axes1]
    
    for i in range(n_frames):
        axes1[i].imshow(values[i][2000:2000+216], cmap='viridis', origin='lower')
        axes1[i].set_title(f'Frame {i}', fontsize=8)
        axes1[i].axis('off')
    
    # Hide unused subplots
    for i in range(n_frames, len(axes1)):
        axes1[i].axis('off')
    
    fig1.suptitle('Hough Accumulators (with squares)')
    plt.tight_layout()
    
    # Create figure for original values
    fig2, axes2 = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    axes2 = axes2.flatten() if rows*cols > 1 else [axes2]
    
    for i in range(n_frames):
        axes2[i].imshow(values_out[i][2000:2000+216], cmap='viridis', origin='lower')
        axes2[i].set_title(f'Frame {i}', fontsize=8)
        axes2[i].axis('off')
    
    for i in range(n_frames, len(axes2)):
        axes2[i].axis('off')
    
    fig2.suptitle('Original Hough Accumulators')
    plt.tight_layout()
    plt.show()
    
    print(f"Displayed {n_frames} frames in {rows}x{cols} grid")
    
    # Print true_peaks for each event in one column
    print("\ntrue_peaks:")
    true_peaks = data["true_peaks"]
    
    for frame_idx, peaks in enumerate(true_peaks):
        print(f"\nFrame {frame_idx} ({len(peaks)} peaks):")
        for peak in peaks:
            print(f"  ({int(peak[0])}, {int(peak[1])})")

# here we display images
display_image_grid("images_short_muons/hough_0.npz", cols=4)