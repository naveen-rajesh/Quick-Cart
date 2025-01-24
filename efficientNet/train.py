import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB0  # Import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.metrics import AUC

# CustomDataGenerator code for multi-label classification
class CustomDataGenerator(Sequence):
    def __init__(self, directory, class_labels_path, img_size=(224, 224), batch_size=32, shuffle=True):  # EfficientNet's default img_size is (224, 224)
        self.directory = directory
        self.img_size = img_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.image_paths = []
        self.labels = []
        self.class_indices = self._load_class_labels(class_labels_path)

        # Load the images and their corresponding labels
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
                        # Append multi-labels as binary vectors
                        label = np.zeros(len(self.class_indices))
                        label[self.class_indices[class_folder]] = 1
                        self.labels.append(label)

        if self.shuffle:
            indices = np.arange(len(self.image_paths))
            np.random.shuffle(indices)
            self.image_paths = [self.image_paths[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))

    def __getitem__(self, index):
        batch_x = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]
        batch_y = self.labels[index * self.batch_size:(index + 1) * self.batch_size]
        x = np.array([preprocess_input(img_to_array(load_img(img_path, target_size=self.img_size))) for img_path in batch_x])
        y = np.array(batch_y)
        return x, y

# Define EfficientNetB0 model architecture for multi-label classification
def create_model(input_shape, num_classes):
    # Load EfficientNetB0 without the top classification layers
    base_model = EfficientNetB0(input_shape=input_shape, include_top=False, weights='imagenet')
    
    # Freeze the base model layers (fine-tuning can be enabled by unfreezing layers later)
    base_model.trainable = False

    # Add classification head for multi-label classification
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='sigmoid')  # Use sigmoid for multi-label classification
    ])

    # Compile the model using binary crossentropy for multi-label classification
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', AUC(name='auc')])
    return model

# Paths
train_dir = r'D:\ML-Dataset\GroceryStoreDataset-master\GroceryStoreDataset-master\dataset\train'
class_labels_path = r'D:\ML-Dataset\class_labels.json'

# Create an instance of the custom data generator
train_gen = CustomDataGenerator(train_dir, class_labels_path)

# Create the model
input_shape = (224, 224, 3)  # EfficientNet expects 224x224x3 input images
num_classes = len(train_gen.class_indices)
model = create_model(input_shape, num_classes)

# Train the model
history = model.fit(train_gen, epochs=10)

# Save the model to a .h5 file
model.save(r'D:\ML-Dataset\trained_model_efficientnet_multilabel.h5')
print("Model saved to trained_model_efficientnet_multilabel.h5")

# Plotting training history
def plot_training_history(history):
    # Get training metrics
    accuracy = history.history['accuracy']
    auc = history.history['auc']
    loss = history.history['loss']
    
    epochs = range(1, len(accuracy) + 1)
   
    # Create subplots
    plt.figure(figsize=(12, 5))
    
    # Plot accuracy
    plt.subplot(1, 3, 1)
    plt.plot(epochs, accuracy, 'bo-', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(1, 3, 2)
    plt.plot(epochs, auc, 'go-', label='Training AUC')
    plt.title('Training AUC')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 3, 3)
    plt.plot(epochs, loss, 'ro-', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # Show plots
    plt.tight_layout()
    plt.show()

# Call the plotting function
plot_training_history(history)
