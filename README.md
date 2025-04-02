# CIFAR-10 Image Classification using CNN

## Overview
This script implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The model is built using TensorFlow and Keras and incorporates techniques like data augmentation, batch normalization, dropout, and learning rate scheduling to improve performance.

## Prerequisites
Ensure you have the following dependencies installed before running the script:

```bash
pip install tensorflow matplotlib numpy
```

Alternatively, if using Conda:
```bash
conda install tensorflow matplotlib numpy
```

## Dataset
The script uses the CIFAR-10 dataset, which consists of 60,000 color images (32x32 pixels) in 10 classes, with 6,000 images per class. The dataset is preloaded using TensorFlow.

## Code Breakdown
### 1. Importing Libraries
```python
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0  # Imported but not used
import matplotlib.pyplot as plt
import numpy as np
```
- TensorFlow and Keras are used to build and train the model.
- Matplotlib is used for visualizing data.
- NumPy is used for numerical operations.

### 2. Loading and Preprocessing Data
```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0
```
- The dataset is loaded and split into training and testing sets.
- Pixel values are normalized to the range [0,1] to improve training stability.

### 3. Displaying Sample Images
```python
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.axis('off')
plt.show()
```
- Displays sample images with class labels for reference.

### 4. Data Augmentation
```python
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])
```
- Randomly modifies images to make the model more robust.

### 5. Defining the CNN Model
```python
def create_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))
    x = data_augmentation(inputs)
    ...
    return tf.keras.Model(inputs=inputs, outputs=outputs)
```
- Defines a CNN model with three convolutional blocks, each followed by batch normalization and dropout.
- Uses `GlobalAveragePooling2D` instead of flattening to reduce overfitting.
- Ends with a softmax activation layer for classification.

### 6. Learning Rate Scheduling
```python
def lr_schedule(epoch):
    if epoch < 5:
        return 3e-4
    elif epoch < 10:
        return 3e-4 * 0.1
    elif epoch < 15:
        return 3e-4 * 0.01
    elif epoch < 20:
        return 3e-4 * 0.001
    else:
        return 3e-4 * 0.0001
```
- Adjusts the learning rate dynamically to improve training stability.

### 7. Compiling and Training the Model
```python
model.compile(optimizer=Adam(learning_rate=3e-4),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
```
- Uses the Adam optimizer and sparse categorical cross-entropy loss.

```python
history = model.fit(train_images, train_labels,
                    epochs=100,
                    batch_size=128,
                    validation_data=(test_images, test_labels),
                    callbacks=[lr_scheduler, early_stop])
```
- Trains the model with validation data and callbacks.

### 8. Visualizing Training Performance
```python
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.show()
```
- Plots accuracy and loss trends during training.

### 9. Evaluating and Saving the Model
```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc:.4f}")
model.save('cifar10_model.h5')
```
- Evaluates the model on the test set and saves it for later use.

### 10. Making Predictions
```python
predictions = model.predict(test_images[:5])
predicted_labels = np.argmax(predictions, axis=1)
```
- Runs the model on test images and displays predictions.

## Performance Results
- Achieved **72% accuracy (0.72)** after **10 epochs**.
- Achieved **89% accuracy (0.89)** after **100 epochs**.
- Both models were trained on a **local machine with 4GB VRAM**.

## Running the Script
Run the script using:
```bash
python cifar10_cnn.py
```

## Conclusion
This script builds a CNN for CIFAR-10 image classification with advanced training techniques. The model achieves good accuracy by using dropout, batch normalization, and learning rate scheduling.

