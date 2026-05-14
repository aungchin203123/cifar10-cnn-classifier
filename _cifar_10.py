
import kagglehub
oxcdcd_cifar10_path = kagglehub.dataset_download('oxcdcd/cifar10')

print('Data source import complete.')

"""### 1. Environment Setup & Data Loading"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, callbacks, optimizers
from tensorflow.keras.utils import image_dataset_from_directory

import warnings
warnings.filterwarnings('ignore')

import os

# Use the path provided by kagglehub
# The dataset structure is usually dataset_root/cifar10/train
train_path = os.path.join(oxcdcd_cifar10_path, 'cifar10', 'train')
test_path = os.path.join(oxcdcd_cifar10_path, 'cifar10', 'test')

# Load training and validation datasets
train_dataset, val_dataset = image_dataset_from_directory(
    train_path,
    image_size=(32, 32),
    validation_split=0.2,
    subset="both",
    seed=42,
    batch_size=32
)

# Define class names for later visualization
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

"""### 2. Dataset Performance Optimization
To ensure the GPU doesn't sit idle while the CPU loads images from Kaggle's disk, we utilize tf.data.AUTOTUNE to dynamically overlap data preprocessing and model execution.
"""

AUTOTUNE = tf.data.AUTOTUNE

# Cache and prefetch datasets to optimize I/O performance
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

"""### 3. Model ArchitectureWe build a robust, VGG-style Sequential model.
The architecture is divided into three feature-extraction blocks,
1. progressively increasing the filter size (32 -> 64 -> 128) while reducing spatial dimensions.  
2. Batch Normalization is used to stabilize and accelerate training.  
3. Dropout is increased progressively to prevent the model from memorizing the training data.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import keras

def build_milestone_model():
    # Data Augmentation Layer (Commented out as requested)
    # data_augmentation = keras.Sequential([
    #     layers.RandomFlip("horizontal"),
    #     layers.RandomRotation(0.1),
    #     layers.RandomZoom(0.1),
    #     layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    #     layers.RandomContrast(factor=0.1)
    # ])

    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        # data_augmentation, # Disabled augmentation

        # Block 1
        # layers.Conv2D(32, (3, 3), activation='relu', padding='same', use_bias=False),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same',kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same',
              kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Classification Head
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_milestone_model()
model.summary()

"""### 12. Advanced Model for 90%+ Accuracy
We will now use a more aggressive augmentation pipeline and a slightly revised architecture to break the 90% barrier.
"""

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers, optimizers

# 1. Advanced Data Augmentation
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
])

def build_optimized_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        data_augmentation,

        # Block 1
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Block 3
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Optimized Head
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

opt_model = build_optimized_model()

# 2. Use CosineDecay for better convergence
initial_learning_rate = 0.001
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate, decay_steps=15625) # Approx 100 epochs

opt_model.compile(
    optimizer=optimizers.Adam(learning_rate=lr_schedule),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Optimized model ready for training.")

# Training the optimized model for more epochs with augmentation
opt_history = opt_model.fit(
    train_dataset,
    epochs=100,
    validation_data=val_dataset,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
    ]
)
print("Training complete.")

"""### 4. Model Training
We compile the model using the Adam optimizer. To squeeze out maximum performance, we implement a ReduceLROnPlateau callback that cuts the learning rate in half if validation accuracy stalls, alongside EarlyStopping to restore the best weights.
"""

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Define Callbacks
lr_reduction = callbacks.ReduceLROnPlateau(monitor='val_accuracy', patience=3, verbose=1, factor=0.5, min_lr=0.00001)
early_stopping = callbacks.EarlyStopping(monitor='val_accuracy', patience=12, restore_best_weights=True)

# Train the model with more epochs and augmentation
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    callbacks=[lr_reduction, early_stopping]
)
print("Training complete.")

"""### 5. Evaluation and Visualization
Now we review the model's learning curves to check for overfitting, followed by a visual inspection of its predictions on the validation set. Finally, we score it against the unseen test directory.
"""

plt.figure(figsize=(10,8))
plt.plot(history.history['loss'],label='train')
plt.plot(history.history['val_loss'],label='val')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# Plot Training vs. Validation Accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(len(acc))

plt.figure(figsize=(8, 6))
plt.plot(epochs_range, acc, label='Training Accuracy', linewidth=2)
plt.plot(epochs_range, val_acc, label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy over Epochs', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Visualizing Predictions
image_batch, label_batch = next(iter(val_dataset))
predictions = model.predict(image_batch, verbose=0)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(12, 12))
for i in range(16):
    ax = plt.subplot(4, 4, i + 1)
    img = image_batch[i].numpy().astype("uint8")
    plt.imshow(img)

    actual = class_names[label_batch[i]]
    pred = class_names[predicted_labels[i]]
    title_color = "green" if actual == pred else "red"

    plt.title(f"Actual: {actual}\nPred: {pred}", color=title_color, fontsize=10)
    plt.axis("off")

plt.tight_layout()
plt.show()

# Final Unbiased Test Evaluation
print("\n--- Final Test Set Evaluation ---")
test_dataset = image_dataset_from_directory(
    test_path,
    image_size=(32, 32),
    batch_size=32
)

results = model.evaluate(test_dataset, verbose=0)
print(f"Final Test Loss: {results[0]:.4f}")
print(f"Final Test Accuracy: {results[1]*100:.2f}%")

"""### 6. Confusion Matrix for Test Set"""

test_dataset_unbatched = image_dataset_from_directory(
    test_path,
    image_size=(32, 32),
    shuffle=False, # Important: Do not shuffle for confusion matrix
    batch_size=32
)

y_pred = model.predict(test_dataset_unbatched)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = tf.concat([y for x, y in test_dataset_unbatched], axis=0)

confusion_mtx = tf.math.confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix for CIFAR-10 Test Set')
plt.show()

"""### 8. Classification Report
Detailed metrics per class to analyze precision, recall, and F1-score.
"""

from sklearn.metrics import classification_report

# Generate the report
report = classification_report(y_true, y_pred_classes, target_names=class_names)
print("Classification Report for CIFAR-10 Test Set:\n")
print(report)

"""### 9. Fine-tuning for Specific Class Improvement
We will continue training with a lower learning rate to help the model better distinguish between visually similar classes like cats and dogs.
"""

# Re-compile with a smaller learning rate for fine-tuning
fine_tune_optimizer = optimizers.Adam(learning_rate=1e-5)

model.compile(
    optimizer=fine_tune_optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train for a few more epochs
fine_tune_history = model.fit(
    train_dataset,
    epochs=10,
    validation_data=val_dataset,
    callbacks=[early_stopping]
)

print("Fine-tuning complete.")

"""### 10. Evaluate Fine-tuning History
Visualizing the loss and accuracy curves specifically for the fine-tuning phase.
"""

# Plot Fine-tuning Results
acc_ft = fine_tune_history.history['accuracy']
val_acc_ft = fine_tune_history.history['val_accuracy']
loss_ft = fine_tune_history.history['loss']
val_loss_ft = fine_tune_history.history['val_loss']

plt.figure(figsize=(12, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(acc_ft, label='Fine-tuning Training Accuracy')
plt.plot(val_acc_ft, label='Fine-tuning Validation Accuracy')
plt.title('Fine-tuning Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(loss_ft, label='Fine-tuning Training Loss')
plt.plot(val_loss_ft, label='Fine-tuning Validation Loss')
plt.title('Fine-tuning Loss')
plt.legend()

plt.show()

"""### 11. Post-Fine-tuning Classification Report
Comparing the metrics after the additional training phase with lower learning rate.
"""

# Predict again using the fine-tuned model
y_pred_ft = model.predict(test_dataset_unbatched)
y_pred_classes_ft = np.argmax(y_pred_ft, axis=1)

# Generate and print the new report
report_ft = classification_report(y_true, y_pred_classes_ft, target_names=class_names)
print("Classification Report After Fine-tuning:\n")
print(report_ft)

"""### 7. Save the Model
Saving the model allows you to reload it later without needing to retrain.
"""

# Save the model in the native Keras format
model.save('cifar10_model.keras')
print("Model saved as cifar10_model.keras")

from google.colab import files

# Download the model file to local machine
files.download('cifar10_model.keras')

import pickle
from google.colab import files

# Save the optimization history object
with open('cifar10_88.pkl', 'wb') as f:
    pickle.dump(history.history, f)

print('History saved as history.pkl. Starting download...')
files.download('cifar10_88.pkl')