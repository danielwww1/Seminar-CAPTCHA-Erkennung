import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import keras.api._v2.keras as keras
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from keras import activations 
from collections import Counter

print("LAUNCHING")
data_dir = 'C:\\Users\\Daniel\\Desktop\\Seminar\\dataset\\onenight\\all'

def load_data(data_dir):
    images = []
    labels = []

    for file in os.listdir(data_dir):
        if file.endswith('.png'):
            image_path = os.path.join(data_dir, file)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (120, 120))
            images.append(image)

            annotation_path = os.path.join(data_dir, os.path.splitext(file)[0] + '.xml')
            tree = ET.parse(annotation_path)
            root = tree.getroot()

            bboxes = []
            for obj in root.findall('object'):
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                bboxes.append([name, xmin, ymin, xmax, ymax])
            labels.append(bboxes)

    return np.array(images), labels

images, labels = load_data(data_dir)


all_class_names = [label[0] for image_labels in labels for label in image_labels]
class_counts = Counter(all_class_names)


classes_to_remove = [class_name for class_name, count in class_counts.items() if count < 700]


filtered_labels = []
for image_labels in labels:
    filtered_image_labels = [label for label in image_labels if label[0] not in classes_to_remove]
    filtered_labels.append(filtered_image_labels)


label_encoder = LabelEncoder()
all_filtered_labels = [label for sublist in filtered_labels for (label, *_) in sublist]
label_encoder.fit(all_filtered_labels)

def encode_labels(labels, max_objects=10):
    encoded_labels = []
    for image_labels in labels:
        encoded_image_labels = np.zeros((max_objects, 5))  # label, xmin, ymin, xmax, ymax 
        for i, (label, xmin, ymin, xmax, ymax) in enumerate(image_labels):
            if i < max_objects:
                encoded_label = label_encoder.transform([label])[0]
                encoded_image_labels[i] = [encoded_label, xmin, ymin, xmax, ymax]
        encoded_labels.append(encoded_image_labels)
    return np.array(encoded_labels)

encoded_labels = encode_labels(filtered_labels)


def flatten_labels(labels):
    flattened = []
    for label_array in labels:
        flattened.append(label_array.flatten())
    return np.array(flattened)

train_images, test_images, train_labels, test_labels = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

train_labels_flat = flatten_labels(train_labels)
test_labels_flat = flatten_labels(test_labels)


label_encoder.fit(all_filtered_labels)


print("Deleted Classes:")
for class_name in classes_to_remove:
    print(class_name)

train_labels_flat = flatten_labels(train_labels)
test_labels_flat = flatten_labels(test_labels)


max_objects = 10  
output_size = max_objects * 5  


all_class_names_filtered = [label[0] for image_labels in filtered_labels for label in image_labels]
class_counts_filtered = Counter(all_class_names_filtered)
print("Anzahl der Begrenzungsrahmen pro Klasse nach dem Filtern:")
for class_name, count in class_counts_filtered.items():
    print(f"{class_name}: {count}")



model = Sequential([
    Conv2D(32, (3, 3), activation='LeakyReLU', input_shape=(120, 120, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(64, (3, 3), activation='LeakyReLU'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.3),

    Conv2D(128, (3, 3), activation='LeakyReLU'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.4),

    Flatten(),
    Dense(256, activation='Softmax'),
    Dropout(0.5),
    Dense(output_size)  
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['accuracy'])


model.summary()


data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)


early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True, verbose=1)


history = model.fit(
    data_gen.flow(train_images, train_labels_flat, batch_size=4),
    steps_per_epoch=len(train_images) // 4,
    epochs=20,  
    validation_data=(test_images, test_labels_flat),
    callbacks=[early_stopping, model_checkpoint]
)


model.save('final_seminarmodel.h5')

print("Model saved.")

# Output label encoder mapping
print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))