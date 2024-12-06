import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

# Download and unzip dataset
import requests, zipfile

url = "https://cdn.freecodecamp.org/project-data/cats-and-dogs/cats_and_dogs.zip"
response = requests.get(url)
with open("cats_and_dogs.zip", "wb") as f:
    f.write(response.content)

with zipfile.ZipFile("cats_and_dogs.zip", "r") as zip_ref:
    zip_ref.extractall(".")

PATH = 'cat-and-dogs'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
test_dir = os.path.join(PATH, 'test')

# Get number of files in each directory. The train and validation directories
# each have the subdirecories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])
total_test = len(os.listdir(test_dir))

# Variables for pre-processing and training.
batch_size = 128
epochs = 15
IMG_HEIGHT = 150
IMG_WIDTH = 150

# 3 -------------------------------------------------------------
train_image_generator = ImageDataGenerator(rescale=1.0 / 255)
validation_image_generator = ImageDataGenerator(rescale=1.0 / 255)
test_image_generator = ImageDataGenerator(rescale=1.0 / 255)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
)
val_data_gen = validation_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=validation_dir,
    shuffle=True,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode='binary',
)

# Filter out invalid files from the DataFrame
valid_files = []
for img_file in os.listdir(test_dir):
    img_path = os.path.join(test_dir, img_file)
    try:
        with Image.open(img_path) as img:
            img.verify()  # Verify the image
            valid_files.append(img_file)  # Add to the list if valid
    except:
        print(f"Skipping invalid file: {img_file}")

# Create DataFrame with valid files
test_df = pd.DataFrame({'filename': [os.path.join(test_dir, f) for f in valid_files]})


# Test data generator using flow_from_dataframe
test_data_gen = test_image_generator.flow_from_dataframe(
    dataframe=test_df,
    x_col='filename',
    y_col=None,  # No labels for the test set
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False,  # Ensures predictions align with the file order
)

# 4 ------------------------------------------------------------------------------
def plotImages(images_arr, probabilities = False):
    fig, axes = plt.subplots(len(images_arr), 1, figsize=(5,len(images_arr) * 3))
    if probabilities is False:
      for img, ax in zip( images_arr, axes):
          ax.imshow(img)
          ax.axis('off')
    else:
      for img, probability, ax in zip( images_arr, probabilities, axes):
          ax.imshow(img)
          ax.axis('off')
          if probability > 0.5:
              ax.set_title("%.2f" % (probability*100) + "% dog")
          else:
              ax.set_title("%.2f" % ((1-probability)*100) + "% cat")
    plt.show()

sample_training_images, _ = next(train_data_gen)
plotImages(sample_training_images[:5])

# 5 ------------------------------------------------------------------------------
train_image_generator = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,         # Randomly rotate images by 0 to 40 degrees
    width_shift_range=0.2,     # Randomly shift images horizontally
    height_shift_range=0.2,    # Randomly shift images vertically
    shear_range=0.2,           # Shear the image randomly
    zoom_range=0.2,            # Randomly zoom into the image
    horizontal_flip=True,      # Randomly flip images horizontally
    fill_mode='nearest',        # Fill in newly created pixels after a transformation
    brightness_range=[0.8, 1.2]  # Adjust brightness between 80% and 120%
)

# 6 ---------------------------------------------------------------------------------------
train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                     directory=train_dir,
                                                     target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                     class_mode='binary')

augmented_images = [train_data_gen[0][0][0] for i in range(5)]

plotImages(augmented_images)

# 7 --------------------------------------------------------------------------------------
model = Sequential([
    # First convolutional layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    # Second convolutional layer
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Third convolutional layer
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Fourth convolutional layer
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    # Flattening the output from the convolutional layers
    Flatten(),

    # Fully connected layer
    Dense(512, activation='relu'),

    # Dropout for regularization
    Dropout(0.5),

    # Output layer with sigmoid activation for binary classification
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(
    optimizer='adam',  # Adaptive Moment Estimation optimizer
    loss='binary_crossentropy',  # Binary classification loss function
    metrics=['accuracy']  # To monitor training and validation accuracy
)

# Display the model's architecture
model.summary()

# 8 --------------------------------------------------------------------------------------
# Calculate steps per epoch
steps_per_epoch = total_train // batch_size
validation_steps = total_val // batch_size

# Train the model
history = model.fit(
    train_data_gen,  # Training data
    steps_per_epoch=steps_per_epoch,  # Number of batches per epoch
    epochs=epochs,  # Number of training epochs
    validation_data=val_data_gen,  # Validation data
    validation_steps=validation_steps  # Number of validation batches per epoch
)

# 9 ----------------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

# Plot training and validation accuracy over epochs
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Plot training and validation loss over epochs
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# 10 --------------------------------------------------------------------------------------

# Get predictions for the test data
probabilities = model.predict(test_data_gen)

# Load all test images (test_data_gen is not shuffled, so all images are sequential)
test_images = []
for i in range(len(test_data_gen)):
    test_images.extend(test_data_gen[i])  # Collect all batches into a single list

# Convert predictions to percentages
probabilities = probabilities.flatten()  # Flatten probabilities to a 1D array
percentages = [prob * 100 if prob > 0.5 else (1 - prob) * 100 for prob in probabilities]

# Plot all test images with their probabilities
plotImages(test_images, probabilities)

# 11 -----------------------------------------------------------------------------
answers =  [1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0,
            1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0,
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1,
            1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1,
            0, 0, 0, 0, 0, 0]

correct = 0

for probability, answer in zip(probabilities, answers):
  if round(probability) == answer:
    correct +=1

percentage_identified = (correct / len(answers)) * 100

passed_challenge = percentage_identified >= 63

print(f"Your model correctly identified {round(percentage_identified, 2)}% of the images of cats and dogs.")

if passed_challenge:
  print("You passed the challenge!")
else:
  print("You haven't passed yet. Your model should identify at least 63% of the images. Keep trying. You will get it!")
