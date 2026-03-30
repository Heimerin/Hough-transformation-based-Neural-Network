import numpy as np
import matplotlib.pyplot as plt
from MC import generate_event, MAGNETIC_CONST

Q_PT_BINS = 256
PHI_BINS = 1024
Q_PT_MIN=-2.0
Q_PT_MAX=2.0
PHI_MIN = 0.0
PHI_MAX = 2*np.pi

q_pt_val = np.linspace(Q_PT_MIN, Q_PT_MAX, Q_PT_BINS)

def fill_hough_accumulator(hits_arr):
    accumulator = np.zeros((Q_PT_BINS, PHI_BINS))

    for hit in hits_arr:
        x_hit, y_hit, layer_r, _ = hit

        phi_hit = np.atan2(y_hit, x_hit)  #do odtworzenia pelnego kąta biegunowego hitu
        if phi_hit < 0:
            phi_hit = phi_hit + 2.0 * np.pi #obrot o 2pi , jesli kat ujemny, zeby miec kat w zakresie [0, 2pi]
        for q_pt_idx, q_pt in enumerate(q_pt_val):
            arg_arcsin = layer_r * MAGNETIC_CONST * q_pt
            if abs(arg_arcsin) <= 1.0:
                phi_0 = phi_hit - np.arcsin(arg_arcsin)
                phi_0 = phi_0 %(2.0 * np.pi) #zawiniecie kata do zakresu [0, 2pi]
                phi_idx = int((phi_0 - PHI_MIN) / (PHI_MAX - PHI_MIN) * PHI_BINS)
                if 0 <= phi_idx < PHI_BINS:
                    accumulator[q_pt_idx, phi_idx] += 1
    return accumulator

def visualize_accumulator(accumulator):
    plt.figure(figsize=(12, 6))

    plt.imshow(accumulator, origin='lower', aspect='auto', cmap='viridis', extent=[PHI_MIN, PHI_MAX, Q_PT_MIN, Q_PT_MAX])
    plt.colorbar(label="Amount of hits")
    plt.xlabel(r"Starting phi $\phi_0$ [rad]")
    plt.ylabel("Curvature $q/p_T$")
    plt.title("Hough Accumulator")
    plt.show()


#do przetestowania transformaty, wygeneruje zdarzenie z pliku od monte carlo i przeprowadzi transformate. 
#ten fragment zadziala tylko wtedy, gdy wlaczymy ten plik jako glowny, a nie bedziemy go importowac do innego skryptu, wiec 
# w przypadku importowania tego pliku do innego skryptu, ten fragment nie zostanie wykonany, dlatego moze tu zostac 
if __name__ == "__main__":
    hit_package, particle_params = generate_event(true_tracks=5, noise_hits=20)
    
    hough_hist = fill_hough_accumulator(hit_package)
    
    visualize_accumulator(hough_hist)