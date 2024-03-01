import numpy as np
from tensorflow.keras.models import load_model

# Example input data
time_stamps = ["00:00:00:01", "00:00:00:02", 
"00:00:00:03", "00:00:00:04", "00:00:00:05",
"00:00:00:06", "00:00:00:010", "00:00:00:013", 
"00:00:00:014", "00:00:00:15", "00:00:00:17", "00:00:00:19"]
heart_rates = np.array([99, 86, 87, 88, 90, 88, 89, 88, 87, 86, 86, 87])

# Normalize the heart rates
min_val, max_val = 50, 220 
normalized_heart_rates = (heart_rates - min_val) / (max_val - min_val)

# Load your model
model_path = "C:\\vignesh\\main_project\\models\\cnn_lstm_hybrid_model.h5"
model = load_model(model_path)

# Create windowed input data for the CNN-LSTM model
window_size = 7  # Should match the window size used during training
X_windowed = []
for i in range(len(normalized_heart_rates)):
    if i < window_size - 1:
        # Pad the sequence with zeros for initial readings
        window = np.zeros(window_size)
        window[-(i+1):] = normalized_heart_rates[:i+1]
    else:
        window = normalized_heart_rates[i - (window_size - 1):i + 1]
    X_windowed.append(window)

X_windowed = np.array(X_windowed).reshape((-1, window_size, 1))

# Predict using the model
y_pred = model.predict(X_windowed)
predicted_labels = (y_pred > 0.5).astype(int).flatten()

# Adjust the first prediction based on the first heart rate value
predicted_labels[0] = 0 if (heart_rates[0]- heart_rates[1]) < 10 else 1

# Output the prediction
print(f"Predicted Labels: {predicted_labels}")
predicted_labels=list(predicted_labels)
if(predicted_labels.count(1))>1:
    print("fault sensor data")
else:
    print("No Fault in the Sensor data")