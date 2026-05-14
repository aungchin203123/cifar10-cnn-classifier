# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.set_page_config(
    page_title="CIFAR-10 Classifier",
    page_icon="🔍",
    layout="wide",
)

CLASS_NAMES_DISPLAY = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck",
]
CLASS_EMOJIS = ["✈️", "🚗", "🐦", "🐱", "🦌", "🐶", "🐸", "🐴", "🚢", "🚛"]


# Load CIFAR-10 dataset and train a simple model on the fly
@st.cache_resource
def get_model():
    with st.spinner("Training model for first time use... (this takes 2-3 minutes)"):
        # Load CIFAR-10 data
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

        # Normalize
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0

        # Simple CNN model
        model = keras.Sequential([
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.3),

            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.4),

            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPooling2D((2, 2)),
            keras.layers.Dropout(0.5),

            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation='softmax')
        ])

        # Compile
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        # Train quickly (just 3 epochs for speed)
        model.fit(x_train, y_train, batch_size=64, epochs=3, validation_split=0.1, verbose=0)

        return model


def preprocess(img):
    img = img.convert("RGB").resize((32, 32))
    return np.expand_dims(np.array(img, np.float32) / 255.0, 0)


def do_predict(model, arr):
    p = model.predict(arr, verbose=0)[0]
    return p, int(np.argmax(p))


# Try to load or train model
try:
    model = get_model()
    st.success("✅ Model ready!")
except Exception as e:
    st.error(f"Model error: {e}")
    model = None

# UI
st.markdown("# CIFAR-10 Image Classifier")
st.markdown("CNN Classifier · 10 classes")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Your image", width=200)
        st.image(image.resize((32, 32)), caption="Resized to 32x32 for model", width=100)

    with col2:
        with st.spinner("Classifying..."):
            probs, idx = do_predict(model, preprocess(image))

        st.markdown(f"## Prediction: {CLASS_EMOJIS[idx]} {CLASS_NAMES_DISPLAY[idx]}")
        st.markdown(f"**Confidence: {probs[idx] * 100:.1f}%**")

        # Show all probabilities
        st.markdown("### All Classes")
        for i in np.argsort(probs)[::-1]:
            pct = probs[i] * 100
            st.progress(pct / 100, text=f"{CLASS_EMOJIS[i]} {CLASS_NAMES_DISPLAY[i]}: {pct:.1f}%")