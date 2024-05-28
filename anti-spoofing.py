import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import os
from tensorflow.keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from PIL import Image

# Define image data generators
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Create data generators for anti-spoofing
train_generator_anti_spoof = train_datagen.flow_from_directory(
    'train_img',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

val_generator_anti_spoof = val_test_datagen.flow_from_directory(
    'test_img',
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Build Anti-Spoofing Model
def build_anti_spoofing_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

input_shape = (128, 128, 3)
anti_spoofing_model = build_anti_spoofing_model(input_shape)

anti_spoofing_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the Anti-Spoofing Model
history = anti_spoofing_model.fit(
    train_generator_anti_spoof,
    validation_data=val_generator_anti_spoof,
    epochs=10,
    callbacks=[ModelCheckpoint('anti_spoofing_model_epoch_{epoch:02d}.h5', save_best_only=False, save_weights_only=False, period=1)]
)

#anti_spoofing_model.save('Anti_Spoofing_Model.h5')
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def evaluate_anti_spoofing_model(model, generator):
    y_true = []
    y_pred = []

    for i in range(len(generator)):
        images, labels = generator[i]
        preds = model.predict(images)
        y_true.extend(labels)
        y_pred.extend(preds)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred_class = (y_pred > 0.5).astype(int)

    accuracy = accuracy_score(y_true, y_pred_class)
    precision = precision_score(y_true, y_pred_class)
    recall = recall_score(y_true, y_pred_class)
    f1 = f1_score(y_true, y_pred_class)
    roc_auc = roc_auc_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred_class)

    print(f'Anti-Spoofing Model Accuracy: {accuracy:.4f}')
    print(f'Anti-Spoofing Model Precision: {precision:.4f}')
    print(f'Anti-Spoofing Model Recall: {recall:.4f}')
    print(f'Anti-Spoofing Model F1 Score: {f1:.4f}')
    print(f'Anti-Spoofing Model ROC AUC Score: {roc_auc:.4f}')
    print('Confusion Matrix:')
    print(cm)

# Load the models for evaluation
classifier_model = tf.keras.models.load_model('ClassifierModel.h5')
embedding_model = tf.keras.models.load_model('Embedding_Model.h5')
anti_spoofing_model = tf.keras.models.load_model('Anti_Spoofing_Model.h5')

# Evaluate the anti-spoofing model
evaluate_anti_spoofing_model(anti_spoofing_model, val_generator_anti_spoof)
