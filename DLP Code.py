import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt

# Path to the dataset
dataset_path = r"E:\DLP\Dataset\train"

# Hyperparameters
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Resize images to 64x64
BATCH_SIZE = 32
EPOCHS = 15

# Step 1: Load dataset
def load_dataset(dataset_path):
    images, labels = [], []
    class_names = sorted(os.listdir(dataset_path))  # Classes are folder names
    for class_name in class_names:
        class_folder = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_folder):
            for img_name in os.listdir(class_folder):
                img_path = os.path.join(class_folder, img_name)
                try:
                    img = Image.open(img_path).convert("RGB")  # Ensure 3 channels
                    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
                    images.append(np.array(img))
                    labels.append(class_name)
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
    return np.array(images), np.array(labels), class_names

images, labels, class_names = load_dataset(dataset_path)

# Step 2: Preprocess the dataset
le = LabelEncoder()
labels_encoded = le.fit_transform(labels)
labels_categorical = to_categorical(labels_encoded)

# Normalize image data
images = images / 255.0

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(images, labels_categorical, test_size=0.2, random_state=42)

# Step 3: Create CNN model
def create_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

model = create_model((IMG_HEIGHT, IMG_WIDTH, 3), len(class_names))

# Step 4: Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=EPOCHS, batch_size=BATCH_SIZE)

# Step 5: Test the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 6: Predict using the model
def predict_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return predicted_class

# Test prediction
sample_image_path = r"C:\Users\Sohan\Downloads\43056.jpg"  # Replace with your test image path
predicted_class = predict_image(model, sample_image_path)
print(f"Predicted Class: {predicted_class}")

# Save the model
model_save_path = r"E:\DLP\traffic_sign_model.keras"  # Specify save location
model.save(model_save_path)
print(f"Model saved to {model_save_path}")