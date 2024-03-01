import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input, Dense, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse
import tensorflow as tf
from sklearn.metrics import accuracy_score

# Assuming you have already loaded the dataset
data = pd.read_csv("/content/updated_heart_rate_dataset_formatted.csv")

# Preprocessing
X = data[['Normalized_Heart_Rate']].values.astype(np.float32)  # Ensure data is float32 to save memory
y = data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# VAE Model
original_dim = X_train.shape[1]
intermediate_dim = 32
latent_dim = 2

# Encoder
inputs = Input(shape=(original_dim,))
h = Dense(intermediate_dim, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_sigma = Dense(latent_dim)(h)

def sampling(args):
    z_mean, z_log_sigma = args
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_sigma) * epsilon

z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_sigma])

# Decoder
decoder_h = Dense(intermediate_dim, activation='relu')
decoder_mean = Dense(original_dim, activation='sigmoid')
h_decoded = decoder_h(z)
x_decoded_mean = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, x_decoded_mean)

# Loss function
xent_loss = original_dim * mse(inputs, x_decoded_mean)
kl_loss = - 0.5 * tf.reduce_sum(1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma), axis=-1)
vae_loss = tf.reduce_mean(xent_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Adjusted batch size to manage memory usage
batch_size = 64  # Reduced batch size

# Train
vae.fit(X_train, X_train,
        epochs=20,
        batch_size=batch_size,
        validation_data=(X_test, X_test))

reconstructed_X_test = vae.predict(X_test, batch_size=batch_size)

# Calculate reconstruction error for each data point
reconstruction_error = np.mean(np.abs(X_test - reconstructed_X_test), axis=1)

# Set a threshold for fault detection
fault_threshold = 0.1

# Predict labels based on reconstruction error
predicted_labels = (reconstruction_error > fault_threshold).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test, predicted_labels)
print(f"VAE Model Accuracy: {accuracy}")