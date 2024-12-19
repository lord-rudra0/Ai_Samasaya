import os
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

def load_data(data_dir):
    images, labels = [], []
    for label in os.listdir(data_dir):
        label_dir = os.path.join(data_dir, label)
        if os.path.isdir(label_dir):
            for img_file in os.listdir(label_dir):
                img_path = os.path.join(label_dir, img_file)
                img = Image.open(img_path).resize((64, 64)).convert("RGB")
                images.append(np.array(img))
                labels.append(label)
    return np.array(images, dtype="float32") / 255.0, np.array(labels)

def prepare_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=(64, 64, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def create_hand_sign_images(text, save_dir, label_to_index, index_to_label, dataset_dir):
    os.makedirs(save_dir, exist_ok=True)
    images = []
    for char in text.upper():
        if char in label_to_index:
            label = index_to_label[label_to_index[char]]
            img_path = os.path.join(dataset_dir, label, os.listdir(os.path.join(dataset_dir, label))[0])
            img = Image.open(img_path)
            img.save(os.path.join(save_dir, f"{char}.png"))
            images.append(img_path)
    return images

def create_video(image_paths, video_path, frame_size=(64, 64), fps=1):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, frame_size)
    for img_path in image_paths:
        if img_path:
            img = cv2.imread(img_path)
            img = cv2.resize(img, frame_size)
            out.write(img)
    out.release()
