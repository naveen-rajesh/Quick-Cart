import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.metrics import classification_report

# CustomDataGenerator code for testing (similar to the training generator)
class TestDataGenerator:
    def __init__(self, directory, class_labels_path, img_size=(224, 224), batch_size=32):  # EfficientNet's default img_size is (224, 224)
        self.directory = directory
        self.img_size = img_size
        self.batch_size = batch_size
        self.image_paths = []
        self.labels = []
        self.class_indices = self._load_class_labels(class_labels_path)
        self._load_data()

    def _load_class_labels(self, class_labels_path):
        with open(class_labels_path, 'r') as f:
            class_labels = json.load(f)
        class_indices = {label: index for index, label in enumerate(class_labels)}
        return class_indices

    def _load_data(self):
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith('.jpg') or file.endswith('.png'):
                    img_path = os.path.join(root, file)
                    self.image_paths.append(img_path)
                    class_folder = os.path.basename(root)
                    if class_folder in self.class_indices:
                        label = np.zeros(len(self.class_indices))
                        label[self.class_indices[class_folder]] = 1
                        self.labels.append(label)

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def get_batch(self, index):
        batch_x = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.array([preprocess_input(img_to_array(load_img(img_path, target_size=self.img_size))) for img_path in batch_x])
        y = np.array(batch_y)
        return x, y

# Paths for the test dataset and trained model
test_dir = r'D:\ML-Dataset\GroceryStoreDataset-master\GroceryStoreDataset-master\dataset\test'
class_labels_path = r'D:\ML-Dataset\class_labels.json'
model_path = r'D:\ML-Dataset\trained_model_efficientnet_multilabel.h5'  # Update this to EfficientNet model

# Load the trained EfficientNet model
model = load_model(model_path)

# Create the test data generator
test_gen = TestDataGenerator(test_dir, class_labels_path)

# Get class labels from file
with open(class_labels_path, 'r') as f:
    class_labels = json.load(f)

# Define a function to make predictions on test data
def evaluate_model(model, test_gen, threshold=0.1):
    y_true = []
    y_pred = []
    
    for i in range(len(test_gen)):
        x_batch, y_batch = test_gen.get_batch(i)
        predictions = model.predict(x_batch)
        
        # Apply the threshold to convert probabilities to binary predictions
        binary_predictions = (predictions >= threshold).astype(int)
        
        y_true.extend(y_batch)
        y_pred.extend(binary_predictions)

    return np.array(y_true), np.array(y_pred)

# Evaluate the model
y_true, y_pred = evaluate_model(model, test_gen)

# Generate classification report
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_labels))

# Function to display predictions for a batch of images
def display_predictions(model, test_gen, batch_index=0, threshold=0.5):
    x_batch, y_batch = test_gen.get_batch(batch_index)
    predictions = model.predict(x_batch)

    for i in range(len(x_batch)):
        actual_labels = [class_labels[j] for j in range(len(y_batch[i])) if y_batch[i][j] == 1]
        predicted_labels = [class_labels[j] for j in range(len(predictions[i])) if predictions[i][j] >= threshold]

        print(f"\nImage {i+1}:")
        print(f"Actual Labels: {', '.join(actual_labels)}")
        print(f"Predicted Labels: {', '.join(predicted_labels)}")

# Display predictions for a batch of test images
display_predictions(model, test_gen, batch_index=0)
