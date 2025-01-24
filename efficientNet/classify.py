import os
import json
import numpy as np
import streamlit as st
import pandas as pd  # Import pandas for tabular display
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input  # Use EfficientNet's preprocess function

# Load class labels and prices from JSON files
def load_class_labels(class_labels_path):
    with open(class_labels_path, 'r') as f:
        class_labels = json.load(f)
    return class_labels

def load_prices(price_mapping_path):
    with open(price_mapping_path, 'r') as f:
        prices = json.load(f)
    return prices

# Preprocess the image for prediction
def preprocess_image(image):
    img_array = img_to_array(image)  # Convert image to numpy array
    img_array = preprocess_input(img_array)  # Preprocess for EfficientNetB0
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Classify the image and return predictions
def classify_image(model, image, class_labels, prices, threshold=0.1):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)[0]  # Shape: (num_classes,)

    # Get the highest prediction score and its corresponding label
    max_index = np.argmax(predictions)
    max_score = predictions[max_index]
    highest_label = class_labels[max_index]

    if max_score < threshold:  # If the highest score is below the threshold
        highest_label = "None"
        max_score = 0  # Reset score

    # Prices for the highest predicted label
    price_for_highest_label = prices.get(highest_label, "Price not available")
    
    return highest_label, max_score, price_for_highest_label

# Load the model and class labels
model_path = r'D:\ML-Dataset\trained_model_efficientnet_multilabel.h5'
class_labels_path = r'D:\ML-Dataset\class_labels.json'
price_mapping_path = r'D:\ML-Dataset\class_mapping.json'

model = load_model(model_path)
class_labels = load_class_labels(class_labels_path)
prices = load_prices(price_mapping_path)

# Streamlit App
st.title("Fruit and Vegetable Image Classifier")

# Upload multiple images
uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    total_price = 0
    bill_summary = {}

    # Create columns based on the number of uploaded files
    cols = st.columns(len(uploaded_files))

    # Classify each uploaded image
    for idx, uploaded_file in enumerate(uploaded_files):
        # Load and display the image in a column
        image = load_img(uploaded_file, target_size=(224, 224))
        with cols[idx]:
            st.image(image, caption='Uploaded Image', use_column_width=True)

            # Classify the image
            highest_label, max_score, price_for_highest_label = classify_image(model, image, class_labels, prices)

            # Display predictions for each image
            st.subheader(f"Highest Prediction:")
            st.write(f"Label: {highest_label}, Score: {max_score:.2f}, Price: {price_for_highest_label}")

            # Update bill summary
            if isinstance(price_for_highest_label, (int, float)):  # Ensure price is a numeric value
                if highest_label in bill_summary:
                    bill_summary[highest_label]['count'] += 1
                    bill_summary[highest_label]['total_price'] += price_for_highest_label
                else:
                    bill_summary[highest_label] = {'count': 1, 'unit_price': price_for_highest_label, 'total_price': price_for_highest_label}

                # Update the total price
                total_price += price_for_highest_label

    # Button to calculate total price
    if st.button("Calculate Total Price"):
        st.subheader("Total Bill:")

        # Create a DataFrame for the bill summary
        bill_df = pd.DataFrame([
            {'Item': label, 'Count': info['count'], 'Unit Price (₹)': info['unit_price'], 'Total Price (₹)': info['total_price']}
            for label, info in bill_summary.items()
        ])

        # Display the bill summary as a table
        st.table(bill_df)

        # Display the total price
        st.write(f"**Total: ₹{total_price:.2f}**")
