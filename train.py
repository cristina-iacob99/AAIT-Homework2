import os
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
from sklearn.model_selection import KFold
from sklearn.ensemble import VotingClassifier
from sklearn.utils import class_weight
import pandas as pd
import tensorflow
from tensorflow.keras.applications.resnet import ResNet50, preprocess_input
from tensorflow.keras.utils import to_categorical
from sklearn.multioutput import MultiOutputClassifier
import matplotlib.pyplot as plt


image_folder_path = './data/task1/'
label_folder_path = './data/task1/task1/train_data/annotations.csv'

image_data = []

# Read the labels from the CSV file
label_df = pd.read_csv(label_folder_path)
label_dict = label_df['label']
labels = label_dict.tolist()
# Iterate through the list of image file names
for image_filename in label_df['sample']:
    # Open the image
    image = cv2.imread(image_folder_path + image_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Append the image data to the list
    image_data.append(image)


image_data = np.array(image_data, dtype = 'float32')
labels = np.array(labels, dtype = 'int32')

image_data = preprocess_input(image_data)
labels = to_categorical(labels)
num_classes = len(labels[0])


with tf.device('/device:GPU:0'):
  input_shape = (64, 64, 3)
  input_tensor = tf.keras.Input(shape=input_shape)

  # Define the ResNet model
  base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape, input_tensor=input_tensor)
  base_model.trainable = False

  # Train the model
  # Define the k-fold cross-validation
  kfold = KFold(n_splits=3, shuffle=True, random_state=42)

  # Initialize the list to store the models
  models = []

  # Perform k-fold cross-validation

  for train, test in kfold.split(image_data, labels):
      
      model = tf.keras.models.Sequential()
      model.add(base_model)
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.Dense(512, activation="relu"))
      model.add(tf.keras.layers.Dropout(0.5))
      model.add(tf.keras.layers.BatchNormalization())
      model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

      # Compile the model
      model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

      # Fit the model on the training data
      history = model.fit(image_data[train], labels[train],validation_data=(image_data[test],labels[test]), batch_size=32, epochs=10, verbose=1)

      # plot the training loss and accuracy
      N = np.arange(0, len(history.history["loss"]))
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(N, history.history["loss"], label="train_loss")
      plt.plot(N, history.history["val_loss"], label="val_loss")
      plt.plot(N, history.history["accuracy"], label="train_accuracy")
      plt.plot(N, history.history["val_accuracy"], label="val_accuracy")
      plt.title("Training Loss and Accuracy [Epoch {}]".format(len(history.history["loss"])))
      plt.xlabel("Epoch Number")
      plt.ylabel("Loss/Accuracy")
      plt.legend()
      
      # save the figure
      plt.savefig('./figs/Task1_training1.png')
      plt.close()
    
      for layer in model.layers[168:]:
        layer.trainable = True
      
      # Fit the model on the training data
      history = model.fit(image_data[train], labels[train],validation_data=(image_data[test],labels[test]), batch_size=32, epochs=50, verbose=1)

      # plot the training loss and accuracy
      N = np.arange(0, len(history.history["loss"]))
      plt.style.use("ggplot")
      plt.figure()
      plt.plot(N, history.history["loss"], label="train_loss")
      plt.plot(N, history.history["val_loss"], label="val_loss")
      plt.plot(N, history.history["accuracy"], label="train_accuracy")
      plt.plot(N, history.history["val_accuracy"], label="val_accuracy")
      plt.title("Training Loss and Accuracy [Epoch {}]".format(len(history.history["loss"])))
      plt.xlabel("Epoch Number")
      plt.ylabel("Loss/Accuracy")
      plt.legend()
      
      # save the figure
      plt.savefig('./figs/Task1_training2.png')
      plt.close()
      # Append the trained model to the list
      models.append(model)

  model.save('models/ResNetKFold_Task1.keras')


# Set the directory path where the validation images are located
val_folder_path = './data/task1/task1/val_data/'
# Get a list of all the validation image file names in the folder
val_filenames = os.listdir(val_folder_path)

# Create a list to store the classification results
classifications = []
val_filenames = sorted(val_filenames)
# Iterate through the list of validation image file names
for val_filename in val_filenames:
    # Open the validation image
    image = cv2.imread(val_folder_path + val_filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.array(image, dtype = 'float32')
    image = preprocess_input(image)
    image = tf.expand_dims(image, 0)
    # Use the model to classify the validation image
    val_predictions = model.predict(image)

    # Get the class with the highest probability
    val_class = np.argmax(val_predictions)
    # Append the classification result to the list
    classifications.append([val_filename, val_class])


classification_df = pd.DataFrame(classifications, columns=['sample', 'label'])
classification_df.to_csv('validation_task1.csv', index=False)