🐟 Fish Classification using CNN and Hyperparameter Tuning

This repository demonstrates the use of a Convolutional Neural Network (CNN) to classify different species of fish from a large-scale dataset. We also perform hyperparameter optimization using Keras Tuner to achieve the best performance.

🔗 View the project on Kaggle: Deep Learning Fish Classification

📑 Table of Contents

📌 Project Overview

📊 Dataset Overview

🏗️ Model Architecture

🔧 Hyperparameter Tuning

📈 Training and Results

📝 Evaluation and Metrics

⚙️ Installation and Setup

🔍 Visualization

🚀 Future Work

📚 References

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Project Overview
The goal of this project is to build a CNN model that can classify images of different fish species. Given the dataset's size and variety, we employ convolutional layers to capture distinguishing features and use Keras Tuner to fine-tune the model's hyperparameters.

Project Features:

CNN Model for image classification.

Hyperparameter Tuning to improve model accuracy.

Visualization of training results and model predictions.
------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

📊 Dataset Overview
The dataset contains images of 9 different species of fish, with 1000 images per class. Each species is well-balanced with equal image distribution, making it suitable for classification tasks.

Fish Species:

Hourse Mackerel

Black Sea Sprat

Sea Bass

Red Mullet

Trout

Striped Red Mullet

Shrimp

Gilt-Head Bream

Red Sea Bream

![Ekran görüntüsü 2024-10-20 221947](https://github.com/user-attachments/assets/27cb2880-f8df-4f8b-b80c-9b4a4b45da8a)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

💡 Dataset Directory Structure:

Fish_Dataset/
│
├── Hourse Mackerel/
├── Black Sea Sprat/
├── Sea Bass/
├── Red Mullet/
├── Trout/
├── Striped Red Mullet/
├── Shrimp/
├── Gilt-Head Bream/
└── Red Sea Bream/

![Ekran görüntüsü 2024-10-20 222007](https://github.com/user-attachments/assets/d16cd3cf-eb7b-4c5b-a512-0605c6f5ec50)

-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🏗️ Model Architecture

The Convolutional Neural Network (CNN) architecture used in this project is designed to capture patterns and features in the fish images. The network consists of:

📐 Layers:

Convolutional Layers: 3 layers with filters increasing progressively from 32 to 64.

MaxPooling Layers: Reduce dimensionality to focus on important features.

Dense Layers: 2 fully connected layers for classification.

Dropout Layer: Prevents overfitting by randomly dropping neurons during training.

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(9, activation='softmax')  # 9 classes for 9 fish species
])
![Ekran görüntüsü 2024-10-20 222023](https://github.com/user-attachments/assets/c6c2c7fc-f7d3-4b92-9def-345352ebc2dd)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🔧 Hyperparameter Tuning

To optimize the model, we used Keras Tuner to search for the best combination of hyperparameters, including:

Number of convolutional layers.

Number of dense layers.

Learning rate.

Dropout rate.

🔍 Best Hyperparameters:
{
    "conv1_units": 96,
    "num_conv_layers": 1,
    "conv2_units": 64,
    "num_dense_layers": 3,
    "dense1_units": 192,
    "dropout1": 0.1,
    "learning_rate": 0.00346,
    "dense2_units": 32,
    "dropout2": 0.0,
    "dense3_units": 32,
    "dropout3": 0.0
}
📈 Training and Results

The model was trained on the dataset for 10 epochs with an 80-20 train-test split.

🏅 Final Model Performance:

Test Accuracy: 87.78%

Test Loss: 0.4121

![Ekran görüntüsü 2024-10-20 222433](https://github.com/user-attachments/assets/f8990e54-d58a-4d82-aa87-5cee417055d7)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



📝 Evaluation and Metrics
The model's performance was evaluated using the following metrics:

Precision
Recall
F1-score
📋 Classification Report:

⚙️ Installation and Setup
Prerequisites
Make sure you have the following installed:

Python 3.7+
TensorFlow 2.x
Keras Tuner
scikit-learn
Matplotlib & Seaborn
Steps to Setup

![Ekran görüntüsü 2024-10-20 222502](https://github.com/user-attachments/assets/152bbcf5-7627-48da-aeaf-93fc43b5f7a5)


🚀 Future Work

Here are some potential improvements for future iterations of this project:

Data Augmentation: Apply data augmentation techniques like rotation, zooming, and flipping to improve model generalization.
Transfer Learning: Experiment with pre-trained models such as ResNet or MobileNet to potentially boost accuracy.
Deeper Architectures: Test deeper architectures or attention mechanisms to improve classification results.

📚 References

TensorFlow Documentation

Keras Tuner Documentation

Kaggle Dataset: Large-Scale Fish Dataset (https://www.kaggle.com/datasets/crowww/a-large-scale-fish-dataset)

View the project on Kaggle: Deep Learning Fish Classification(https://www.kaggle.com/code/emirtunal/deep-learning-fish)
