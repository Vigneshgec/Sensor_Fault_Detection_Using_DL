import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import joblib

# Load the dataset
dataset = pd.read_csv(r'c:\vignesh\main_project\updated_heart_rate_dataset_formatted.csv')

# Normalize the Heart_Rate column
scaler = MinMaxScaler(feature_range=(0, 1))
dataset['Heart_Rate'] = scaler.fit_transform(dataset[['Heart_Rate']])

# Convert the dataset into a format suitable for CNN (time series)
def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)        
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 10  # Number of time steps to look back
X, y = create_dataset(dataset[['Heart_Rate']], dataset['Label'], time_steps)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Reshape input for CNN [samples, time_steps, features]
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, shuffle=False)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"CNN Test Accuracy: {test_accuracy}")
# Save the model and scaler
#model.save(r'C:\vignesh\main_project\models\cnn_model.h5')
#joblib.dump(scaler, r'C:\vignesh\main_project\models\cnn_scaler.pkl')