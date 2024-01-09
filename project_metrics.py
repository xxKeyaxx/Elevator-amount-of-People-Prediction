import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv('dataset.csv')

# Step 2: Preprocess the data
image_paths = df['Path'].tolist()
labels = df['Number_of_People'].tolist()

model = YOLO('yolov8m.pt')
prediction = []

i = 0
for image_path in image_paths:
    result = model.predict(source = image_path, classes = [0], conf = 0.3)
    cls = result[0].boxes.cls
    detections = cls.tolist().count(0)
    # # for doing tolerance = 1
    # if abs(detections - labels[i]) == 1:
    #     detections = labels[i]
    # i = i + 1
    prediction.append(detections)


accuracy = accuracy_score(labels, prediction)
precision = precision_score(labels, prediction, average='weighted', zero_division=1)
recall = recall_score(labels, prediction, average='weighted', zero_division=1)
weighted_f1 = f1_score(labels, prediction, average='weighted', zero_division=1)


print("Classification Metrics:")
print("Accuracy:", accuracy)
print("Weighted-average Precision:", precision)
print("Weighted-average Recall:", recall)
print("Weighted-average F1-score:", weighted_f1)

print()
print("Regression Metrics:")

# Compute Mean Absolute Error (MAE)
mae = mean_absolute_error(labels, prediction)
print(f"Mean Absolute Error (MAE): {mae}")

# Compute Mean Squared Error (MSE)
mse = mean_squared_error(labels, prediction)
print(f"Mean Squared Error (MSE): {mse}")

# Compute Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Compute R^2 score
r2 = r2_score(labels, prediction)
print(f"R^2 Score: {r2}")

confusion_matrix = confusion_matrix(labels, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.show()