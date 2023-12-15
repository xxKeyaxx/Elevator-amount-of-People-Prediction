import pandas as pd
from ultralytics import YOLO
import supervision as sv
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from PIL import Image

# Step 1: Load the CSV file into a DataFrame
df = pd.read_csv('dataset.csv')

# Step 2: Preprocess the data
image_paths = df['Path'].tolist()
labels = df['Number_of_People'].tolist()

model = YOLO('yolov8m.pt')
prediction = []

i = 0
for image_path in image_paths:
    image = Image.open(image_path);
    result = model(image)
    result = list(result)[0]
    detections = sv.Detections.from_yolov8(result)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.3)]
    detections = len(detections)
    # # for doing tolerance = 1
    # if abs(detections - labels[i]) == 1:
    #     detections = labels[i]
    # i = i + 1
    prediction.append(detections)

accuracy = accuracy_score(labels, prediction)
precision = precision_score(labels, prediction, average='weighted', zero_division=1)
recall = recall_score(labels, prediction, average='weighted', zero_division=1)
weighted_f1 = f1_score(labels, prediction, average='weighted', zero_division=1)

print("Accuracy:", accuracy)
print("Weighted-average Precision:", precision)
print("Weighted-average Recall:", recall)
print("Weighted-average F1-score:", weighted_f1)

confusion_matrix = confusion_matrix(labels, prediction)
disp = ConfusionMatrixDisplay(confusion_matrix)
disp.plot()
plt.show()