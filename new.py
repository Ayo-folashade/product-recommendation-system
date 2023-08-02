import tensorflow as tf
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

# Load the pre-trained ResNet50 model
model = tf.keras.applications.resnet50.ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the model's weights, so they won't be updated during training
model.trainable = False

# Create a new model that appends a GlobalMaxPooling2D layer after the ResNet50 model
model = tf.keras.Sequential([
    model,
    tf.keras.layers.GlobalMaxPooling2D()
])


# Extract features from an image using the pre-trained model
def extract_features(img_path, model):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    expanded_img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = tf.keras.applications.resnet50.preprocess_input(expanded_img_array)
    result = model.predict(preprocessed_img).flatten()
    normalized_result = result / norm(result)

    return normalized_result


# Create a list of file paths for the images in the 'subset_images' folder
filenames = []
for file in os.listdir('subset_images'):
    filenames.append('subset_images/' + file)

# Extract features from each image and store them in a list
feature_list = []
for file in tqdm(filenames):
    feature_list.append(extract_features(file, model))

# Save the extracted features and filenames as pickle files
pickle.dump(feature_list, open('embeddings.pkl', 'wb'))
pickle.dump(filenames, open('filenames.pkl', 'wb'))
