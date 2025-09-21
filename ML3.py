# Step 1: Import libraries
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

images = []
labels = []

image_size = (64, 64)

# Step 2: Set dataset path
dataset_path = r"C:\Users\sonia\OneDrive\Documents\Projects\train"  # <-- your actual folder


print("Files found in folder:", os.listdir(dataset_path))  # Debug: list all files

for file_name in os.listdir(dataset_path):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
        if "dog" in file_name.lower():
            label = 1
        elif "cat" in file_name.lower():
            label = 0
        else:
            print(f"Skipping file (no label match): {file_name}")
            continue

        img_path = os.path.join(dataset_path, file_name)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Failed to load {img_path}")
            continue

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, image_size)
        images.append(img.flatten())
        labels.append(label)

print("Total images loaded:", len(images))