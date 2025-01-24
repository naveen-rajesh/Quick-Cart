import os
import json
import numpy as np
import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input

base_dir = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_dir, 'trained_model_efficientnet_multilabel.h5')
class_labels_path = os.path.join(base_dir, 'class_labels.json')
price_mapping_path = os.path.join(base_dir, 'class_mapping.json')

def load_class_labels(class_labels_path):
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels

def load_prices(price_mapping_path):
    with open(price_mapping_path, 'r') as f:
        prices = json.load(f)
    return prices

def preprocess_image(image):
    img_array = img_to_array(image)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(model, image, class_labels, prices, threshold=0.1):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]

    max_index = np.argmax(predictions)
    max_score = predictions[max_index]
    highest_label = class_labels[max_index]

    if max_score < threshold:
        highest_label = "None"
        max_score = 0

    price_for_highest_label = prices.get(highest_label, "Price not available")
    
    return highest_label, max_score, price_for_highest_label

model = load_model(model_path)
class_labels = load_class_labels(class_labels_path)
prices = load_prices(price_mapping_path)

st.title("Fruit and Vegetable Image Classifier")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    total_price = 0
    bill_summary = {}

    cols = st.columns(len(uploaded_files))

    for idx, uploaded_file in enumerate(uploaded_files):
        image = load_img(uploaded_file, target_size=(224, 224))
        with cols[idx]:
            st.image(image, caption='Uploaded Image', use_column_width=True)

            highest_label, max_score, price_for_highest_label = classify_image(model, image, class_labels, prices)

            st.subheader(f"Highest Prediction:")
            st.write(f"Label: {highest_label}, Score: {max_score:.2f}, Price: {price_for_highest_label}")

            if isinstance(price_for_highest_label, (int, float)):
                if highest_label in bill_summary:
                    bill_summary[highest_label]['count'] += 1
                    bill_summary[highest_label]['total_price'] += price_for_highest_label
                else:
                    bill_summary[highest_label] = {'count': 1, 'unit_price': price_for_highest_label, 'total_price': price_for_highest_label}

                total_price += price_for_highest_label

    if st.button("Calculate Total Price"):
        st.subheader("Total Bill:")

        bill_df = pd.DataFrame([
            {'Item': label, 'Count': info['count'], 'Unit Price (₹)': info['unit_price'], 'Total Price (₹)': info['total_price']}
            for label, info in bill_summary.items()
        ])

        st.table(bill_df)

        st.write(f"**Total: ₹{total_price:.2f}**")
