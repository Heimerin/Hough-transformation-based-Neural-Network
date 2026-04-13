import numpy as np 
import tensorflow as tf
import matplotlib.pyplot as plt

from MC import generate_event
from hough import fill_hough_accumulator, Q_PT_BINS, PHI_BINS
from count_analysis import gen_truth_heatmap, build_unet

NUM_TRAIN_EVENTS = 2000
NUM_VAL_EVENTS = 400
BATCH_SIZE = 8
EPOCHS = 10

def hough_data_generator(num_events, noise_hits=50):
    """
    Generator do produkcji par (X,Y) w locie, gdzie X jest akumulatorem
    a Y jest mapą cieplną Gaussa
    """
    for i in range(num_events):
        n_tracks = np.random.randint(2, 12) #do wyboru ile czastek prawdziwych będzie
        hits, params = generate_event(true_tracks=n_tracks, noise_hits=noise_hits)
        #transformata i etykieta:
        accumulator = fill_hough_accumulator(hits)
        heatmap=gen_truth_heatmap(params)
        X = np.expand_dims(accumulator, axis=-1).astype(np.float32)
        Y = np.expand_dims(heatmap, axis=-1).astype(np.float32)

        yield X, Y

output_signature = (tf.TensorSpec(shape=(Q_PT_BINS, PHI_BINS, 1), dtype=tf.float32),
                    tf.TensorSpec(shape=(Q_PT_BINS, PHI_BINS, 1), dtype=tf.float32)
                    
)

train_dataset = tf.data.Dataset.from_generator(
    lambda: hough_data_generator(NUM_TRAIN_EVENTS),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_generator(
    lambda: hough_data_generator(NUM_VAL_EVENTS),
    output_signature=output_signature
).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

if __name__ == "__main__":
    model = build_unet()
    history = model.fit(train_dataset, validation_data = val_dataset, epochs=EPOCHS)

    model.save('unet_hough.keras')
    plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('MSE')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)
    plt.show()