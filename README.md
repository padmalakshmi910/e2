# Developing a Neural Network Classification Model

## AIM
To develop a neural network classification model for the given dataset.

## THEORY
The Iris dataset consists of 150 samples from three species of iris flowers (Iris setosa, Iris versicolor, and Iris virginica). Each sample has four features: sepal length, sepal width, petal length, and petal width. The goal is to build a neural network model that can classify a given iris flower into one of these three species based on the provided features.

## Neural Network Model
Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Data Collection and Understanding – Load the dataset, inspect features, and identify the target variable.

### STEP 2:
Data Cleaning and Encoding – Handle missing values and convert categorical data and labels into numerical form.

### STEP 3:
Feature Scaling and Data Splitting – Normalize features and split data into training and testing sets.

### STEP 4:
Model Architecture Design – Define the neural network layers, neurons, and activation functions.

### STEP 5:
Model Training and Optimization – Train the model using a loss function and optimizer through backpropagation.

### STEP 6:
Model Evaluation and Prediction – Evaluate performance using metrics and make predictions on unseen data.


## PROGRAM

~~~
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torch.utils.data import TensorDataset, DataLoader

import seaborn as sns
import matplotlib.pyplot as plt

# Load Dataset
data = pd.read_csv("customers (1).csv")

# Display dataset
print(data.head())
print(data.columns)

# Drop ID column
data = data.drop(columns=["ID"])

# Handle missing values
data.fillna({
    "Work_Experience": 0,
    "Family_Size": data["Family_Size"].median()
}, inplace=True)

# Encode categorical columns
categorical_columns = [
    "Gender",
    "Ever_Married",
    "Graduated",
    "Profession",
    "Spending_Score",
    "Var_1"
]

for col in categorical_columns:
    data[col] = LabelEncoder().fit_transform(data[col])

# Encode target variable
label_encoder = LabelEncoder()
data["Segmentation"] = label_encoder.fit_transform(data["Segmentation"])

# Split features and target
X = data.drop(columns=["Segmentation"])
y = data["Segmentation"].values

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Feature Scaling
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to Tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)

y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# Neural Network
class PeopleClassifier(nn.Module):
    def __init__(self, input_size):
        super(PeopleClassifier, self).__init__()

        self.fc1 = nn.Linear(input_size, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 8)
        self.fc4 = nn.Linear(8, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Training Function
def train_model(model, train_loader, criterion, optimizer, epochs):

    model.train()

    for epoch in range(epochs):

        for inputs, labels in train_loader:

            optimizer.zero_grad()

            outputs = model(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Initialize Model
model = PeopleClassifier(input_size=X_train.shape[1])

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train Model
train_model(model, train_loader, criterion, optimizer, epochs=100)

# Evaluation
model.eval()

predictions = []
actuals = []

with torch.no_grad():

    for X_batch, y_batch in test_loader:

        outputs = model(X_batch)

        _, predicted = torch.max(outputs, 1)

        predictions.extend(predicted.numpy())
        actuals.extend(y_batch.numpy())

# Performance Metrics
accuracy = accuracy_score(actuals, predictions)

conf_matrix = confusion_matrix(actuals, predictions)

class_report = classification_report(
    actuals,
    predictions,
    target_names=[str(i) for i in label_encoder.classes_]
)

print("Name: G.PADMA LAKSHMI")
print("Register No: 212225230206")
print(f"Test Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)
print("Classification Report:")
print(class_report)

# Confusion Matrix Plot
plt.figure(figsize=(6, 5))

sns.heatmap(
    conf_matrix,
    annot=True,
    cmap="Blues",
    fmt="g",
    xticklabels=label_encoder.classes_,
    yticklabels=label_encoder.classes_
)

plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")

plt.show()

# Prediction
sample_input = X_test[12].unsqueeze(0)

with torch.no_grad():
    output = model(sample_input)

predicted_class_index = torch.argmax(output, dim=1).item()

predicted_class_label = label_encoder.inverse_transform(
    [predicted_class_index]
)[0]

actual_class_label = label_encoder.inverse_transform(
    [y_test[12].item()]
)[0]

print("Name: G.PADMA LAKSHMI")
print("Register No: 212225230206")
print(f"Predicted Class: {predicted_class_label}")
print(f"Actual Class: {actual_class_label}")

~~~

### Name:G.Padma Lakshmi

### Register Number:212225230206


### Dataset Information

![alt text](<Screenshot 2026-07-30 211250.png>)

### OUTPUT

![alt text](<Screenshot 2026-07-30 211229.png>)
![alt text](<Screenshot 2026-07-30 211148.png>)
![alt text](<Screenshot 2026-07-30 211159.png>)


### New Sample Data Prediction

![alt text](<Screenshot 2026-07-30 213256.png>)

## RESULT

The neural network model was trained successfully and customer segments were predicted.