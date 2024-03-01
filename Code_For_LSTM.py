import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load the dataset
data = pd.read_csv("C:\\Users\\vigne\\Downloads\\updated_heart_rate_dataset_formatted.csv")

# Assuming 'Normalized_Heart_Rate' is the feature and 'Label' is the target
X = data[['Normalized_Heart_Rate']].values
y = data['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# LSTM Model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=100, validation_split=0.1, verbose=1, callbacks=[EarlyStopping(monitor='val_loss', patience=10)])

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"LSTM Test Accuracy: {test_accuracy}")

# Save
#model.save(r'C:\vignesh\main_project\models\lstm_model2.h5')
