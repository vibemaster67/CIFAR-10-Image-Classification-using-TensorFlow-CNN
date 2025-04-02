# Install TensorFlow if not already installed
# conda install tensorflow

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import EfficientNetB0  # Imported but not used—oops, forgot to remove it!
import matplotlib.pyplot as plt
import numpy as np

# Load the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

# Normalize the images to 0-1 range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Quick check of the data shapes
print(f"Training images shape: {train_images.shape}")
print(f"Training labels shape: {train_labels.shape}")
print(f"Test images shape: {test_images.shape}")
print(f"Test labels shape: {test_labels.shape}")

# Define class names for CIFAR-10 (handy for visualization later)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Let's see some sample images
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(train_images[i])
    plt.title(class_names[train_labels[i][0]])
    plt.axis('off')
plt.show()

# Set up data augmentation to make the model more robust
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),    # Flip images horizontally
    layers.RandomRotation(0.1),         # Slight rotation
    layers.RandomZoom(0.1),             # Random zoom
    layers.RandomContrast(0.1),         # Adjust contrast
])

# Define the CNN model
def create_model():
    inputs = tf.keras.Input(shape=(32, 32, 3))  # Input shape for CIFAR-10 images
    x = data_augmentation(inputs)               # Apply augmentation
    
    # First block of conv layers
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)          # Normalize to stabilize training
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)          # Downsample
    x = layers.Dropout(0.3)(x)                  # Prevent overfitting
    
    # Second block
    x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.4)(x)
    
    # Third block
    x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling2D()(x)      # Reduce to a vector
    
    # Dense layers for classification
    x = layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)  # 10 classes for CIFAR-10
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create the model
model = create_model()

# Learning rate schedule (manually defined steps)
def lr_schedule(epoch):
    if epoch < 5:
        return 3e-4         # Start with this
    elif epoch < 10:
        return 3e-4 * 0.1   # Drop a bit
    elif epoch < 15:
        return 3e-4 * 0.01  # Drop more
    elif epoch < 20:
        return 3e-4 * 0.001 # Even lower
    else:
        return 3e-4 * 0.0001 # Really low now

# Set up callbacks
early_stop = EarlyStopping(monitor='val_accuracy', patience=15, restore_best_weights=True)
lr_scheduler = LearningRateScheduler(lr_schedule)

# Compile the model with Adam optimizer
model.compile(optimizer=Adam(learning_rate=3e-4),
              loss='sparse_categorical_crossentropy',  # Using sparse since labels are integers
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels,
                    epochs=100,
                    batch_size=128,
                    validation_data=(test_images, test_labels),
                    callbacks=[lr_scheduler, early_stop])

# Plot the training progress
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate on test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"Test accuracy: {test_acc:.4f}")

# Save the model for later use
model.save('cifar10_model.h5')
print("Model saved to cifar10_model.h5")

# Let's predict on a few test images and display them
num_images = 5
test_images_sample = test_images[:num_images]
predictions = model.predict(test_images_sample)
predicted_labels = np.argmax(predictions, axis=1)

plt.figure(figsize=(15, 3))
for i in range(num_images):
    plt.subplot(1, num_images, i+1)
    plt.imshow(test_images_sample[i])
    plt.title(f"Pred: {class_names[predicted_labels[i]]}\nTrue: {class_names[test_labels[i][0]]}")
    plt.axis('off')
plt.show()