# Sensor_Fault_Detection_Using_DL
Sensor Data Integrity with Deep Learning-Based Fault Detection
Overview
This repository contains the implementation and research findings of a project focused on advancing sensor data integrity through deep learning-based fault detection methodologies.

Abstract
In the realm of modern data-driven systems, accurate and reliable sensor measurements are imperative for informed decision-making and system integrity. This project aims to develop a robust sensor fault detection methodology leveraging deep learning techniques. By harnessing the power of deep learning, this project presents a tangible solution to accurately identify faulty sensor data in real-time, bolstering the dependability and efficacy of sensor-driven systems.

Methodology
The methodology section outlines a comprehensive approach that includes:

Data Collection and Preprocessing
Strategic Model Selection and Development (LSTM, CNN, RNN, VAE)
In-depth Training and Optimization Process
Rigorous Testing and Validation for Model Efficacy
Dataset Overview and Preprocessing
Model Architecture
Results and Experimental Evaluation
The results section discusses the findings of the research, including:

Performance of LSTM and RNN Models
Efficacy of CNN and VAE Models
RNN Model Testing with Accurate and Faulty Data
Implications and Future Applications
Repository Structure
data/: Contains the dataset used for training and testing.
models/: Includes the implementation of different deep learning models (LSTM, CNN, RNN, VAE).
scripts/: Contains scripts for data preprocessing, model training, and evaluation.
figures/: Contains visualizations used in the research findings.
README.md: Provides an overview of the project, methodology, results, and repository structure.
How to Use
Clone the repository:

git clone https://github.com/your-username/sensor-data-fault-detection.git
Navigate to the repository directory:

cd sensor-data-fault-detection
Install dependencies:

pip install -r requirements.txt
Explore the scripts/ directory for data preprocessing, model training, and evaluation scripts.

Utilize the models/ directory for implementing and experimenting with different deep learning architectures.

Refer to the data/ directory for the dataset used in the research.

Process to execute the main hybrid code, you'll need to follow these steps:
Install necessary packages: Ensure you have numpy and tensorflow installed. You can install them using pip if you haven't already:

pip install numpy tensorflow
Save your model: Make sure you have a trained model saved in the specified path (cnn_lstm_hybrid_model.h5). This model should be trained to accept input data in the format defined in the code.

Run the code: Copy the provided code into a Python script or a Jupyter Notebook. Ensure the path to your model file is correct. Then execute the script or notebook.

Understand the code flow:

The code begins by importing necessary libraries (numpy and load_model from tensorflow.keras.models).
Example input data (time_stamps and heart_rates) are provided. heart_rates are normalized to be between 0 and 1.
The trained model is loaded from the specified path.
The input data is windowed to match the input shape expected by the model. The window size is set to 7.
The model predicts the labels for the input data.
The first prediction is adjusted based on the difference between the first two heart rate values.
Finally, the predicted labels are outputted, and based on the count of fault predictions, a message indicating whether there's a fault in the sensor data is printed.
Review the output: After execution, the script will print the predicted labels and a message indicating whether there's a fault in the sensor data based on the predictions.

By following these steps, you should be able to execute the provided code and analyze its output. Make sure your environment is properly set up with the required packages and that the model file exists in the specified path.
